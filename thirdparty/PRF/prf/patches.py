from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from prf.config import PRFConfig
from prf.io_gs import covariance_world
from prf.types import GaussianScene, SurfacePatch


def select_patch_anchors(scene: GaussianScene, cfg: PRFConfig) -> np.ndarray:
    """One highest-opacity Gaussian per voxel, then cap to max_anchors."""
    keep = scene.opacity >= cfg.min_opacity
    mu = scene.mu[keep]
    opacity = scene.opacity[keep]
    source = np.nonzero(keep)[0]
    if mu.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    voxel = max(float(cfg.anchor_voxel_m), 1e-6)
    keys = np.floor(mu / voxel).astype(np.int64)
    order = np.argsort(-opacity, kind="mergesort")
    _, first = np.unique(keys[order], axis=0, return_index=True)
    anchors = source[order[first]]
    if anchors.size > cfg.max_anchors:
        ranked = np.argsort(-scene.opacity[anchors])
        anchors = anchors[ranked[: cfg.max_anchors]]
    return anchors.astype(np.int64)


def build_surface_patch(
    scene: GaussianScene,
    tree: cKDTree,
    anchor_id: int,
    patch_id: int,
    cfg: PRFConfig,
) -> SurfacePatch:
    """Fit one patch on the anchor's orientation cluster.

    Production fields use ``build_candidate_patches_at_anchor`` so a crease
    can emit both surfaces. This helper keeps the single-cluster path for
    unit tests and diagnostics that inspect the anchor-owned plane.
    """
    neighborhoods = _neighborhood_member_sets(scene, tree, anchor_id, cfg, all_clusters=False)
    if not neighborhoods:
        return _invalid_patch(scene.mu[anchor_id], patch_id)
    chosen = neighborhoods[0]
    for item in neighborhoods:
        if int(anchor_id) in set(item["member_ids"].tolist()):
            chosen = item
            break
    return assemble_patch_from_members(scene, chosen, patch_id, cfg)


def build_candidate_patches_at_anchor(
    scene: GaussianScene,
    tree: cKDTree,
    anchor_id: int,
    cfg: PRFConfig,
) -> list[SurfacePatch]:
    """Every orientation cluster × spatial component with enough support."""
    neighborhoods = _neighborhood_member_sets(
        scene, tree, anchor_id, cfg, all_clusters=bool(cfg.split_all_orientation_clusters)
    )
    patches = []
    for index, item in enumerate(neighborhoods):
        patch = assemble_patch_from_members(scene, item, index, cfg)
        if patch.valid:
            patches.append(patch)
    return patches


def deduplicate_patches(patches: list[SurfacePatch], cfg: PRFConfig) -> list[SurfacePatch]:
    """Drop near-duplicate candidates by center, unsigned normal, and member IoU."""
    if not patches:
        return []
    ranked = sorted(
        patches,
        key=lambda patch: (
            not patch.geometry_valid,
            -float(patch.inlier_fraction),
            -int(patch.ids.size),
            float(patch.eigen_ratio),
        ),
    )
    kept: list[SurfacePatch] = []
    member_sets = []
    min_cos = float(np.cos(np.deg2rad(cfg.dedup_normal_deg)))
    for patch in ranked:
        members = set(np.asarray(patch.ids, dtype=np.int64).tolist())
        duplicate = False
        for other, other_members in zip(kept, member_sets):
            if abs(float(np.dot(patch.normal, other.normal))) < min_cos:
                continue
            scale = cfg.dedup_center_radius_scale * min(float(patch.radius), float(other.radius))
            if float(np.linalg.norm(patch.center - other.center)) > max(scale, cfg.delta_min_m):
                continue
            union = len(members | other_members)
            iou = (len(members & other_members) / union) if union else 0.0
            if iou >= cfg.dedup_member_iou:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(patch)
        member_sets.append(members)
    out = []
    for patch_id, patch in enumerate(kept):
        patch.patch_id = int(patch_id)
        out.append(patch)
    return out


def assemble_patch_from_members(
    scene: GaussianScene,
    neighborhood: dict,
    patch_id: int,
    cfg: PRFConfig,
) -> SurfacePatch:
    """Two-pass weighted PCA on one candidate member set."""
    member_ids = np.asarray(neighborhood["member_ids"], dtype=np.int64)
    nbr_ids = np.asarray(neighborhood["neighborhood_ids"], dtype=np.int64)
    k_query = int(neighborhood["k_query"])
    clustered = bool(neighborhood["clustered"])
    cluster_fraction = float(neighborhood["cluster_fraction"])
    second_cluster_fraction = float(neighborhood["second_cluster_fraction"])
    mixed = bool(neighborhood["mixed"])

    fit = fit_plane_two_pass(scene, member_ids, cfg, clustered=clustered)
    ids = fit["ids"]
    center = fit["center"]
    normal = fit["normal"]
    t1 = fit["t1"]
    t2 = fit["t2"]
    tangent = fit["tangent"]
    eigen_ratio = float(fit["eigen_ratio"])
    inlier_fraction = float(fit["inlier_fraction"])
    thickness_m = float(fit["thickness_m"])
    plane_residual_m = float(fit["plane_residual_m"])
    keep_fraction = float(ids.size) / float(max(k_query, 1))

    flags: list[str] = []
    valid = ids.size >= cfg.min_patch_gaussians
    if not valid:
        flags.append("LOW_SUPPORT")
    if mixed:
        flags.append("MIXED_NEIGHBORHOOD")
    if eigen_ratio > cfg.nonplanar_eigen_ratio:
        flags.append("NONPLANAR")
    if nbr_ids.size and float(np.median(scene.opacity[nbr_ids])) < cfg.min_opacity:
        flags.append("LOW_OPACITY_PATCH")

    if ids.size:
        surface_coords = (scene.mu[ids] - center) @ tangent
        radius = float(np.median(np.linalg.norm(surface_coords, axis=1)))
        radius = max(radius, cfg.delta_min_m)
        area_m2, hull_uv = convex_hull_uv(surface_coords)
    else:
        radius = cfg.delta_min_m
        area_m2 = 0.0
        hull_uv = np.zeros((0, 2), dtype=np.float64)
    hull_margin = hull_interior_margin(hull_uv)
    if hull_margin < cfg.border_hull_margin:
        flags.append("BORDER")

    hull_ok = hull_uv.shape[0] >= 3 and area_m2 > 0.0
    geometry_valid = bool(
        valid
        and inlier_fraction >= cfg.geometry_min_inlier_fraction
        and eigen_ratio <= cfg.nonplanar_eigen_ratio
        and hull_ok
        and hull_margin > cfg.geometry_severe_border_margin
        and "LOW_OPACITY_PATCH" not in flags
    )
    geometry_confidence = _geometry_confidence(
        inlier_fraction, eigen_ratio, hull_margin, int(ids.size), geometry_valid, cfg
    )
    return SurfacePatch(
        patch_id=patch_id,
        center=center,
        normal=normal,
        t1=t1,
        t2=t2,
        tangent=tangent,
        ids=ids,
        radius=radius,
        valid=valid,
        area_m2=area_m2,
        hull_uv=hull_uv,
        quality_flags=flags,
        thickness_m=thickness_m,
        eigen_ratio=eigen_ratio,
        keep_fraction=keep_fraction,
        cluster_fraction=cluster_fraction,
        inlier_fraction=inlier_fraction,
        second_cluster_fraction=second_cluster_fraction,
        hull_margin=hull_margin,
        geometry_valid=geometry_valid,
        geometry_confidence=geometry_confidence,
        plane_residual_m=plane_residual_m,
        support_count=int(ids.size),
    )


def fit_plane_two_pass(
    scene: GaussianScene,
    member_ids: np.ndarray,
    cfg: PRFConfig,
    *,
    clustered: bool,
) -> dict:
    """Weighted PCA, thickness inliers, then PCA again on the inliers."""
    ids0 = np.asarray(member_ids, dtype=np.int64)
    if ids0.size == 0:
        zaxis = np.array([0.0, 0.0, 1.0])
        return {
            "ids": ids0,
            "center": np.zeros(3),
            "normal": zaxis,
            "t1": np.array([1.0, 0.0, 0.0]),
            "t2": np.array([0.0, 1.0, 0.0]),
            "tangent": np.eye(3)[:, :2],
            "eigen_ratio": 0.0,
            "inlier_fraction": 0.0,
            "thickness_m": 0.0,
            "plane_residual_m": 0.0,
        }
    center, normal, t1, t2, eigval, delta = _weighted_pca_frame(scene, ids0)
    sigma_n = _normal_scales(scene, ids0, normal)
    tau_perp = cfg.k_perp * float(np.median(sigma_n)) if sigma_n.size else 0.0
    keep = np.abs(delta @ normal) < max(tau_perp, cfg.delta_min_m)
    if cfg.use_gaussian_normals and scene.normals is not None and not clustered:
        keep &= np.abs(scene.normals[ids0] @ normal) > np.cos(np.deg2rad(cfg.theta_surface_deg))
    ids = ids0[keep]
    inlier_fraction = float(ids.size) / float(max(ids0.size, 1))
    if ids.size >= 3:
        center, normal, t1, t2, eigval, delta_in = _weighted_pca_frame(scene, ids)
        residual_delta = delta_in
    else:
        residual_delta = (scene.mu[ids] - center) if ids.size else np.zeros((0, 3))
    eigen_ratio = float(eigval[0] / eigval[1]) if eigval[1] > 1e-12 else 0.0
    if residual_delta.size:
        plane_residual_m = float(np.sqrt(np.mean((residual_delta @ normal) ** 2)))
    else:
        plane_residual_m = 0.0
    tangent = np.column_stack((t1, t2))
    return {
        "ids": ids,
        "center": center,
        "normal": normal,
        "t1": t1,
        "t2": t2,
        "tangent": tangent,
        "eigen_ratio": eigen_ratio,
        "inlier_fraction": inlier_fraction,
        "thickness_m": float(tau_perp),
        "plane_residual_m": plane_residual_m,
    }


def _weighted_pca_frame(scene: GaussianScene, ids: np.ndarray):
    points = scene.mu[ids]
    alpha = np.maximum(scene.opacity[ids], 1e-12)
    weights = alpha / np.maximum(alpha.sum(), 1e-12)
    center = weights @ points
    delta = points - center
    cov = (delta * weights[:, None]).T @ delta
    eigval, eigvec = np.linalg.eigh(cov)
    normal = eigvec[:, 0]
    t1 = eigvec[:, 1]
    t2 = eigvec[:, 2]
    if scene.normals is not None and ids.size:
        ref = _unsigned_mean_normal(scene.normals[ids])
        if float(np.dot(normal, ref)) < 0.0:
            normal = -normal
            t1 = -t1
    return center, normal, t1, t2, eigval, delta


def _neighborhood_member_sets(
    scene: GaussianScene,
    tree: cKDTree,
    anchor_id: int,
    cfg: PRFConfig,
    *,
    all_clusters: bool,
) -> list[dict]:
    k = min(int(cfg.patch_knn), scene.num_gaussians)
    _, nbr_ids = tree.query(scene.mu[anchor_id], k=k)
    nbr_ids = np.atleast_1d(np.asarray(nbr_ids, dtype=np.int64))
    mu = scene.mu[nbr_ids]
    anchor_local = int(np.where(nbr_ids == anchor_id)[0][0]) if np.any(nbr_ids == anchor_id) else 0
    cluster_fraction = 1.0
    second_cluster_fraction = 0.0
    mixed = False
    clustered = False
    local_groups: list[np.ndarray]

    if cfg.cluster_normals and cfg.use_gaussian_normals and scene.normals is not None:
        clustered = True
        cluster_ids, sizes = unsigned_normal_clusters(
            scene.normals[nbr_ids], np.cos(np.deg2rad(cfg.theta_surface_deg)), seed=anchor_local
        )
        ranked = sorted(sizes, reverse=True)
        second_cluster_fraction = float(ranked[1]) / float(max(k, 1)) if len(ranked) > 1 else 0.0
        if all_clusters:
            local_groups = []
            for cid in range(len(sizes)):
                oriented = np.nonzero(cluster_ids == cid)[0]
                if oriented.size < cfg.min_patch_gaussians:
                    continue
                local_groups.extend(
                    spatial_components(
                        mu, oriented, k_connect=cfg.spatial_connect_knn, min_size=cfg.min_patch_gaussians
                    )
                )
            anchor_oriented = np.nonzero(cluster_ids == cluster_ids[anchor_local])[0]
            cluster_fraction = float(anchor_oriented.size) / float(max(k, 1))
        else:
            oriented = np.nonzero(cluster_ids == cluster_ids[anchor_local])[0]
            cluster_fraction = float(oriented.size) / float(max(k, 1))
            connected = spatial_component(
                mu, oriented, anchor_local, k_connect=cfg.spatial_connect_knn
            )
            local_groups = [connected if connected.size >= cfg.min_patch_gaussians else oriented]
        mixed = cluster_fraction < cfg.mixed_cluster_fraction or second_cluster_fraction >= cfg.mixed_second_fraction
    else:
        whole = np.arange(nbr_ids.size, dtype=np.int64)
        if all_clusters:
            local_groups = spatial_components(
                mu, whole, k_connect=cfg.spatial_connect_knn, min_size=cfg.min_patch_gaussians
            )
            if not local_groups:
                local_groups = [whole]
        else:
            local_groups = [whole]

    out = []
    for local_ids in local_groups:
        if local_ids.size == 0:
            continue
        out.append(
            {
                "member_ids": nbr_ids[np.asarray(local_ids, dtype=np.int64)],
                "neighborhood_ids": nbr_ids,
                "k_query": k,
                "clustered": clustered,
                "cluster_fraction": cluster_fraction,
                "second_cluster_fraction": second_cluster_fraction,
                "mixed": mixed,
            }
        )
    return out


def _geometry_confidence(
    inlier_fraction: float,
    eigen_ratio: float,
    hull_margin: float,
    support_count: int,
    geometry_valid: bool,
    cfg: PRFConfig,
) -> float:
    planar = max(0.0, 1.0 - float(eigen_ratio) / max(float(cfg.nonplanar_eigen_ratio), 1e-6))
    interior = min(max(float(hull_margin), 0.0) / max(float(cfg.border_hull_margin), 1e-6), 1.0)
    support = min(float(support_count) / float(max(cfg.min_patch_gaussians * 2, 1)), 1.0)
    value = float(inlier_fraction) * planar * max(interior, 0.25) * max(support, 0.25)
    if not geometry_valid:
        value *= 0.25
    return float(np.clip(value, 0.0, 1.0))


def _invalid_patch(center: np.ndarray, patch_id: int) -> SurfacePatch:
    zaxis = np.array([0.0, 0.0, 1.0])
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([0.0, 1.0, 0.0])
    return SurfacePatch(
        patch_id=patch_id,
        center=np.asarray(center, dtype=np.float64).copy(),
        normal=zaxis,
        t1=t1,
        t2=t2,
        tangent=np.column_stack((t1, t2)),
        ids=np.zeros((0,), dtype=np.int64),
        radius=1e-3,
        valid=False,
        quality_flags=["LOW_SUPPORT"],
        support_count=0,
    )


def unsigned_normal_clusters(normals: np.ndarray, min_cos: float, *, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Greedy sign-invariant clusters: each cluster is |n · n_seed| >= min_cos."""
    vectors = np.asarray(normals, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    n = vectors.shape[0]
    used = np.zeros(n, dtype=bool)
    cluster_ids = np.full(n, -1, dtype=np.int32)
    sizes: list[int] = []
    order = [int(seed)] + [i for i in range(n) if i != int(seed)]
    cid = 0
    for start in order:
        if used[start]:
            continue
        members = np.where(~used & (np.abs(vectors @ vectors[start]) >= min_cos))[0]
        cluster_ids[members] = cid
        used[members] = True
        sizes.append(int(members.size))
        cid += 1
    return cluster_ids, sizes


def spatial_component(points: np.ndarray, local_ids: np.ndarray, anchor_local: int, k_connect: int) -> np.ndarray:
    """Keep the kNN-graph component of the selected cluster that contains the anchor."""
    local_ids = np.asarray(local_ids, dtype=np.int64)
    if local_ids.size == 0:
        return local_ids
    if int(anchor_local) not in set(local_ids.tolist()):
        return local_ids
    if local_ids.size <= max(int(k_connect), 1) + 1:
        return local_ids
    pts = np.asarray(points, dtype=np.float64)[local_ids]
    k = min(int(k_connect) + 1, pts.shape[0])
    _, nn = cKDTree(pts).query(pts, k=k)
    parent = np.arange(pts.shape[0], dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, row in enumerate(np.atleast_2d(nn)):
        for j in np.atleast_1d(row):
            union(i, int(j))
    anchor_pos = int(np.where(local_ids == int(anchor_local))[0][0])
    root = find(anchor_pos)
    keep = np.array([find(i) == root for i in range(pts.shape[0])])
    return local_ids[keep]


def spatial_components(
    points: np.ndarray, local_ids: np.ndarray, k_connect: int, min_size: int
) -> list[np.ndarray]:
    """All kNN-graph connected components of a cluster with at least min_size members."""
    local_ids = np.asarray(local_ids, dtype=np.int64)
    if local_ids.size < int(min_size):
        return []
    if local_ids.size <= max(int(k_connect), 1) + 1:
        return [local_ids]
    pts = np.asarray(points, dtype=np.float64)[local_ids]
    k = min(int(k_connect) + 1, pts.shape[0])
    _, nn = cKDTree(pts).query(pts, k=k)
    parent = np.arange(pts.shape[0], dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, row in enumerate(np.atleast_2d(nn)):
        for j in np.atleast_1d(row):
            union(i, int(j))
    groups: dict[int, list[int]] = {}
    for i in range(pts.shape[0]):
        groups.setdefault(find(i), []).append(i)
    return [
        local_ids[np.asarray(members, dtype=np.int64)]
        for members in groups.values()
        if len(members) >= int(min_size)
    ]


def _unsigned_mean_normal(normals: np.ndarray) -> np.ndarray:
    vectors = np.asarray(normals, dtype=np.float64)
    ref = vectors[0]
    aligned = vectors * np.sign(np.einsum("ij,j->i", vectors, ref) + 1e-12)[:, None]
    mean = aligned.mean(axis=0)
    nrm = float(np.linalg.norm(mean))
    if nrm <= 1e-12:
        return ref / max(float(np.linalg.norm(ref)), 1e-12)
    return mean / nrm


def hull_interior_margin(hull_uv: np.ndarray) -> float:
    """Distance from the origin to the hull boundary, divided by the circumradius."""
    hull = np.asarray(hull_uv, dtype=np.float64).reshape(-1, 2)
    if hull.shape[0] < 3:
        return 0.0
    if not _origin_inside_convex(hull):
        return 0.0
    max_r = max(float(np.linalg.norm(hull, axis=1).max()), 1e-12)
    closed = np.vstack((hull, hull[0]))
    distances = [
        _point_to_segment_distance(np.zeros(2), closed[i], closed[i + 1]) for i in range(hull.shape[0])
    ]
    return float(min(distances) / max_r)


def _origin_inside_convex(hull: np.ndarray) -> bool:
    for i in range(hull.shape[0]):
        a = hull[i]
        b = hull[(i + 1) % hull.shape[0]]
        cross = (b[0] - a[0]) * (-a[1]) - (b[1] - a[1]) * (-a[0])
        if cross < -1e-12:
            return False
    return True


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    span = end - start
    denom = float(np.dot(span, span))
    if denom <= 1e-18:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, span) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


def _normal_scales(scene: GaussianScene, ids: np.ndarray, normal: np.ndarray) -> np.ndarray:
    sigma = scene.scales[ids]
    rotations = scene.rotations[ids]
    cov = covariance_world(rotations, sigma)
    return np.sqrt(np.maximum(np.einsum("i,nij,j->n", normal, cov, normal), 0.0))


def convex_hull_uv(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Tangent-plane convex hull. 2D ConvexHull.volume is the enclosed area."""
    points = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        return 0.0, np.zeros((0, 2), dtype=np.float64)
    if points.shape[0] < 3:
        return 0.0, points.copy()
    try:
        hull = ConvexHull(points)
    except QhullError:
        return 0.0, points.copy()
    return float(hull.volume), points[hull.vertices].copy()


def pack_csr(blocks: list[np.ndarray], *, last_axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate variable-length arrays. offsets[i]:offsets[i+1] is block i."""
    offsets = np.zeros(len(blocks) + 1, dtype=np.int32)
    if not blocks:
        if last_axis:
            return offsets, np.zeros((0, last_axis), dtype=np.float64)
        return offsets, np.zeros((0,), dtype=np.int64)
    arrays = [np.asarray(block) for block in blocks]
    lengths = np.array([int(arr.shape[0]) for arr in arrays], dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    nonempty = [arr for arr in arrays if arr.shape[0]]
    if nonempty:
        return offsets, np.concatenate(nonempty, axis=0)
    return offsets, arrays[0][:0].copy()


def hull_world_points(patch: SurfacePatch) -> np.ndarray:
    hull = np.asarray(patch.hull_uv, dtype=np.float64).reshape(-1, 2)
    if hull.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return patch.center + hull[:, 0:1] * patch.t1 + hull[:, 1:2] * patch.t2


def sample_points(patch: SurfacePatch) -> np.ndarray:
    hull = hull_world_points(patch)
    if hull.shape[0] == 0:
        return np.asarray(patch.center, dtype=np.float64).reshape(1, 3)
    return np.vstack((np.asarray(patch.center, dtype=np.float64).reshape(1, 3), hull))


def patches_from_archive(archive: dict[str, np.ndarray]) -> list[SurfacePatch]:
    """Rebuild SurfacePatch objects from a schema-v2 NPZ archive."""
    patch_ids = np.asarray(archive["patch_id"])
    member_offsets = np.asarray(archive["member_offsets"], dtype=np.int64)
    member_ids = np.asarray(archive["member_gaussian_ids"], dtype=np.int64)
    hull_offsets = np.asarray(archive["hull_offsets"], dtype=np.int64)
    hull_uv = np.asarray(archive["hull_uv"], dtype=np.float64).reshape(-1, 2)
    flags = np.asarray(archive["quality_flags"]).astype(str)
    geometry_valid = np.asarray(archive["geometry_valid"], dtype=np.uint8) if "geometry_valid" in archive else None
    geometry_confidence = (
        np.asarray(archive["geometry_confidence"], dtype=np.float64) if "geometry_confidence" in archive else None
    )
    plane_residual = (
        np.asarray(archive["plane_residual_m"], dtype=np.float64) if "plane_residual_m" in archive else None
    )
    support_count = np.asarray(archive["support_count"], dtype=np.int64) if "support_count" in archive else None
    patches: list[SurfacePatch] = []
    for i, patch_id in enumerate(patch_ids.tolist()):
        t1 = np.asarray(archive["tangent_1"][i], dtype=np.float64)
        t2 = np.asarray(archive["tangent_2"][i], dtype=np.float64)
        ids = member_ids[member_offsets[i] : member_offsets[i + 1]]
        raw_flags = [name for name in flags[i].split("|") if name]
        patches.append(
            SurfacePatch(
                patch_id=int(patch_id),
                center=np.asarray(archive["center"][i], dtype=np.float64),
                normal=np.asarray(archive["normal"][i], dtype=np.float64),
                t1=t1,
                t2=t2,
                tangent=np.column_stack((t1, t2)),
                ids=ids.copy(),
                radius=float(archive["patch_radius"][i]),
                valid=True,
                area_m2=float(archive["patch_area"][i]),
                hull_uv=hull_uv[hull_offsets[i] : hull_offsets[i + 1]].copy(),
                quality_flags=raw_flags,
                geometry_valid=bool(geometry_valid[i]) if geometry_valid is not None else False,
                geometry_confidence=float(geometry_confidence[i]) if geometry_confidence is not None else 0.0,
                plane_residual_m=float(plane_residual[i]) if plane_residual is not None else 0.0,
                support_count=int(support_count[i]) if support_count is not None else int(ids.size),
                inlier_fraction=float(archive["inlier_fraction"][i]) if "inlier_fraction" in archive else 0.0,
                eigen_ratio=float(archive["eigen_ratio"][i]) if "eigen_ratio" in archive else 0.0,
            )
        )
    return patches
