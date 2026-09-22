"""Train-only overlapping ROI coverage for joint LoD. Test views never enter."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from lod.eval_protocol import (
    LOD_GAIN_BASELINE,
    MASK_POLICY,
    OVERLAP_WINDOWS,
    PROTOCOL,
    native_crop_box,
)
from lod.jax068 import (
    GRID_OVERLAP,
    MIN_MASK_FRAC,
    MIN_WORLD_DIST_FRAC,
    NEIGHBOR_K,
    PROBE_TILES,
    ROI_MARKER,
    TEST_EVAL_WINDOW_CENTERS,
    TEST_VIEWS,
    VOXEL_BINS,
)
from utils.zoom_camera import NormalizedROI


SINGLE_ROI_BUDGET = {"steps": 500, "max_points": 50000}


def overlapping_centers(zoom: float, overlap: float = GRID_OVERLAP) -> tuple[float, ...]:
    """Centers of overlapping zoom windows that still fit in [0, 1]."""

    if not (0.0 < float(overlap) < 1.0):
        raise ValueError(f"overlap must be in (0, 1), got {overlap}")
    visible = 1.0 / float(zoom)
    half = 0.5 * visible
    step = visible * (1.0 - float(overlap))
    end = 1.0 - half
    if step <= 1e-9:
        return (0.5,)
    values: list[float] = []
    x = half
    while x <= end + 1e-9:
        values.append(round(min(x, end), 6))
        x += step
    if not values or abs(values[-1] - end) > 1e-5:
        values.append(round(end, 6))
    unique: list[float] = []
    for value in values:
        if all(abs(value - prev) > 1e-8 for prev in unique):
            unique.append(value)
    return tuple(unique)


def tile_id(view_name: str, zoom: float, center_x: float, center_y: float) -> str:
    return f"{view_name}__z{float(zoom):g}__{float(center_x):.3f}_{float(center_y):.3f}"


def tile_roi(tile: Mapping[str, Any]) -> NormalizedROI:
    return NormalizedROI(
        float(tile["center_x"]),
        float(tile["center_y"]),
        float(tile.get("width", ROI_MARKER)),
        float(tile.get("height", ROI_MARKER)),
    )


def mask_frac_in_box(mask_hw, box: Sequence[int]) -> float:
    left, top, right, bottom = (int(v) for v in box)
    crop = mask_hw[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    keep = crop > 0.5
    return float(keep.mean()) if hasattr(keep, "mean") else float(sum(keep.reshape(-1)) / keep.size)


def candidate_grid(
    *,
    view_name: str,
    width: int,
    height: int,
    mask_hw,
    zoom: float = 2.0,
    overlap: float = GRID_OVERLAP,
    min_mask_frac: float = MIN_MASK_FRAC,
) -> list[dict[str, Any]]:
    """2x (or given zoom) overlapping windows on one train view's valid mask."""

    rows: list[dict[str, Any]] = []
    for cy in overlapping_centers(zoom, overlap):
        for cx in overlapping_centers(zoom, overlap):
            roi = NormalizedROI(cx, cy, ROI_MARKER, ROI_MARKER)
            if not roi.zoom_is_valid(zoom) or not roi.zoom_is_valid(4.0):
                continue
            box = native_crop_box(width, height, roi, zoom)
            frac = mask_frac_in_box(mask_hw, box) if mask_hw is not None else 1.0
            if frac < float(min_mask_frac):
                continue
            rows.append(
                {
                    "tile_id": tile_id(view_name, zoom, cx, cy),
                    "view_name": view_name,
                    "zoom": float(zoom),
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "width": ROI_MARKER,
                    "height": ROI_MARKER,
                    "box": [int(v) for v in box],
                    "mask_frac": float(frac),
                    "xyz": None,
                    "zoom_4x_nested": True,
                }
            )
    return rows


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _aabb(points: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    origin = [min(p[i] for p in points) for i in range(3)]
    corner = [max(p[i] for p in points) for i in range(3)]
    return origin, [c - o for o, c in zip(origin, corner)]


def _voxel(xyz: Sequence[float], origin: Sequence[float], size: Sequence[float], n_bins: int) -> tuple[int, int, int]:
    idx = []
    for i in range(3):
        span = max(float(size[i]), 1e-8)
        rel = (float(xyz[i]) - float(origin[i])) / span
        idx.append(min(int(n_bins) - 1, max(0, int(rel * int(n_bins)))))
    return int(idx[0]), int(idx[1]), int(idx[2])


def greedy_select_3d(
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_world_dist: float,
    voxel_bins: int = VOXEL_BINS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep spatially spread tiles, then fill voxels that still have candidates.

    Stops multiple cameras from stacking on one building while leaving other
    3D cells empty. Tiles without xyz are dropped.
    """

    ranked = sorted(
        (dict(item) for item in candidates if item.get("xyz") is not None),
        key=lambda item: (-float(item["mask_frac"]), item["view_name"], item["center_x"], item["center_y"]),
    )
    selected: list[dict[str, Any]] = []
    for item in ranked:
        xyz = item["xyz"]
        if all(_dist(xyz, keep["xyz"]) >= float(min_world_dist) for keep in selected):
            item = dict(item)
            item["select_reason"] = "spread"
            selected.append(item)

    occupancy: dict[str, Any] = {
        "min_world_dist": float(min_world_dist),
        "voxel_bins": int(voxel_bins),
        "n_candidates_with_xyz": len(ranked),
        "n_after_spread": len(selected),
        "missed_voxels": [],
        "coverage_frac": 1.0,
    }
    if len(ranked) < 2:
        occupancy["n_selected"] = len(selected)
        return selected, occupancy

    origin, size = _aabb([item["xyz"] for item in ranked])
    occupancy["aabb_origin"] = origin
    occupancy["aabb_size"] = size
    candidate_voxels: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for item in ranked:
        candidate_voxels.setdefault(_voxel(item["xyz"], origin, size, voxel_bins), []).append(item)
    selected_voxels = {_voxel(item["xyz"], origin, size, voxel_bins) for item in selected}
    missed = sorted(set(candidate_voxels) - selected_voxels)
    selected_ids = {item["tile_id"] for item in selected}
    for voxel in missed:
        extra = max(candidate_voxels[voxel], key=lambda item: float(item["mask_frac"]))
        if extra["tile_id"] in selected_ids:
            continue
        extra = dict(extra)
        extra["select_reason"] = "missed_voxel"
        selected.append(extra)
        selected_ids.add(extra["tile_id"])
        selected_voxels.add(voxel)
    occupancy["missed_voxels"] = [list(v) for v in missed]
    occupancy["n_candidate_voxels"] = len(candidate_voxels)
    occupancy["n_selected_voxels"] = len(selected_voxels)
    occupancy["coverage_frac"] = len(selected_voxels) / max(len(candidate_voxels), 1)
    occupancy["n_selected"] = len(selected)
    return selected, occupancy


def farthest_probe_ids(tiles: Sequence[Mapping[str, Any]], k: int = PROBE_TILES) -> list[str]:
    """Lock a small spatially spread subset for the joint implementation probe."""

    usable = [item for item in tiles if item.get("xyz") is not None]
    if not usable:
        return [str(item["tile_id"]) for item in tiles[:k]]
    chosen = [max(usable, key=lambda item: float(item["mask_frac"]))]
    while len(chosen) < min(int(k), len(usable)):
        def score(item: Mapping[str, Any]) -> float:
            if any(item["tile_id"] == keep["tile_id"] for keep in chosen):
                return -1.0
            return min(_dist(item["xyz"], keep["xyz"]) for keep in chosen)

        nxt = max(usable, key=score)
        if score(nxt) < 0:
            break
        chosen.append(nxt)
    return [str(item["tile_id"]) for item in chosen]


def nearest_neighbor_names(
    view_name: str,
    centers: Mapping[str, Sequence[float]],
    *,
    k: int = NEIGHBOR_K,
) -> list[str]:
    others = [name for name in centers if name != view_name]
    others.sort(key=lambda name: _dist(centers[view_name], centers[name]))
    return others[: max(0, int(k))]


def primary_tiles_by_view(tiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One stage camera per L0 name: first (highest-mask) tile of that view."""

    ranked = sorted(
        tiles,
        key=lambda item: (-float(item.get("mask_frac", 0.0)), item["view_name"], item["center_x"], item["center_y"]),
    )
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in ranked:
        name = str(item["view_name"])
        if name in seen:
            continue
        seen.add(name)
        out.append(dict(item))
    return out


def image_union_coverage(mask_hw, boxes: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Valid-mask fraction covered by the union of native crop boxes."""

    import numpy as np

    mask = np.asarray(mask_hw)
    if mask.ndim == 3:
        mask = mask[0]
    valid = mask > 0.5
    union = np.zeros(valid.shape, dtype=bool)
    for box in boxes:
        left, top, right, bottom = (int(v) for v in box)
        union[top:bottom, left:right] = True
    n_valid = int(valid.sum())
    n_covered = int((valid & union).sum())
    return {
        "n_valid": n_valid,
        "n_covered": n_covered,
        "n_uncovered": n_valid - n_covered,
        "valid_coverage": float(n_covered / max(n_valid, 1)),
    }


def footprint_occupancy(
    candidate_points: Sequence[Sequence[Sequence[float]]],
    selected_points: Sequence[Sequence[Sequence[float]]],
    *,
    voxel_bins: int = VOXEL_BINS,
) -> dict[str, Any]:
    """3D occupancy of footprints (many samples per tile), not just crop centers."""

    flat_c = [xyz for samples in candidate_points for xyz in samples if xyz is not None]
    if len(flat_c) < 2:
        return {"coverage_frac": 1.0, "n_candidate_voxels": 0, "n_selected_voxels": 0, "missed_voxels": []}
    origin, size = _aabb(flat_c)
    cand = set()
    for samples in candidate_points:
        for xyz in samples:
            if xyz is not None:
                cand.add(_voxel(xyz, origin, size, voxel_bins))
    selected = set()
    for samples in selected_points:
        for xyz in samples:
            if xyz is not None:
                selected.add(_voxel(xyz, origin, size, voxel_bins))
    missed = sorted(cand - selected)
    return {
        "kind": "footprint_not_center",
        "voxel_bins": int(voxel_bins),
        "aabb_origin": origin,
        "aabb_size": size,
        "n_candidate_samples": len(flat_c),
        "n_candidate_voxels": len(cand),
        "n_selected_voxels": len(selected),
        "missed_voxels": [list(v) for v in missed],
        "coverage_frac": len(selected & cand) / max(len(cand), 1),
    }


def view_supervision_stats(
    voxel_views: Mapping[tuple[int, int, int], Sequence[str]],
) -> dict[str, Any]:
    """Spatial coverage is not view coverage. Record which cameras see each voxel."""

    n = len(voxel_views)
    n_views = [len(set(names)) for names in voxel_views.values()]
    single = sum(1 for count in n_views if count == 1)
    return {
        "n_voxels": n,
        "n_single_view_voxels": single,
        "n_multi_view_voxels": n - single,
        "views_per_voxel_min": min(n_views) if n_views else 0,
        "views_per_voxel_median": float(sorted(n_views)[len(n_views) // 2]) if n_views else 0.0,
        "views_per_voxel_max": max(n_views) if n_views else 0,
        "do_not_add_dropped_cameras_for_view_coverage": True,
    }


def visit_stats(
    sample_counts: Mapping[str, int],
    *,
    skip_keys: Sequence[str] = ("__train_1x__",),
    expected: int | None = None,
) -> dict[str, Any]:
    values = [int(count) for key, count in sample_counts.items() if key not in set(skip_keys)]
    ordered = sorted(values)
    n = len(ordered)
    return {
        "sampling": "random_with_replacement",
        "expected_visits_per_tile": None if expected is None else int(expected),
        "expected_is_not_guaranteed": True,
        "n_tiles": n,
        "min": ordered[0] if ordered else 0,
        "median": ordered[n // 2] if ordered else 0,
        "max": ordered[-1] if ordered else 0,
        "n_unvisited": sum(1 for value in ordered if value == 0),
        "note": (
            "Random tile draws make 960*0.8/64=12 an expectation under uniform "
            "sampling, not a completed round-robin cover. Strict 12 rounds would "
            "need a shuffled per-tile schedule."
        ),
    }


def nested_4x_tiles(
    tiles_2x: Sequence[Mapping[str, Any]],
    *,
    width: int = 2048,
    height: int = 2048,
) -> list[dict[str, Any]]:
    """Same centers at 4x. The 4x crop is smaller than the parent 2x crop."""

    rows: list[dict[str, Any]] = []
    for item in tiles_2x:
        roi = tile_roi(item)
        if not roi.zoom_is_valid(4.0):
            continue
        row = dict(item)
        row["zoom"] = 4.0
        row["parent_2x"] = item["tile_id"]
        row["tile_id"] = tile_id(item["view_name"], 4.0, item["center_x"], item["center_y"])
        row["box"] = [int(v) for v in native_crop_box(width, height, roi, 4.0)]
        rows.append(row)
    return rows


def locked_eval_windows(
    test_views: Sequence[str] = TEST_VIEWS,
    *,
    width: int = 2048,
    height: int = 2048,
) -> list[dict[str, Any]]:
    """Test-side windows. Fixed before training; overlapping, not a success rate."""

    rows: list[dict[str, Any]] = []
    for view in test_views:
        for window_id, cx, cy in TEST_EVAL_WINDOW_CENTERS:
            roi = NormalizedROI(float(cx), float(cy), ROI_MARKER, ROI_MARKER)
            for zoom in (2.0, 4.0):
                if not roi.zoom_is_valid(zoom):
                    raise ValueError(f"locked eval window {window_id} is not valid at {zoom:g}x")
                box = native_crop_box(width, height, roi, zoom)
                rows.append(
                    {
                        "window_id": f"{view}__{window_id}__z{zoom:g}",
                        "image_name": view,
                        "split": "test",
                        "roi_id": window_id,
                        "zoom": float(zoom),
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "width": ROI_MARKER,
                        "height": ROI_MARKER,
                        "box": [int(v) for v in box],
                    }
                )
    return rows


def joint_budget(
    n_tiles: int,
    *,
    mix_ratio: float = 0.2,
    visits_per_tile: int = 12,
    probe: bool = False,
    n_probe_tiles: int = PROBE_TILES,
) -> dict[str, Any]:
    """Coverage rounds + total steps. Point cap follows coverage, not the 50k copy."""

    mix = float(mix_ratio)
    if not (0.0 < mix < 1.0):
        raise ValueError(f"mix_ratio must be in (0, 1), got {mix_ratio}")
    if probe:
        used = min(int(n_probe_tiles), max(int(n_tiles), 0))
        return {
            "mode": "joint_probe",
            "n_tiles_used": used,
            "mix_ratio": mix,
            "steps": 40,
            "l2_freeze_steps": 10,
            "max_points": 8000,
            "densify_from": 1,
            "densify_until": 30,
            "densify_interval": 10,
            "densify_grad_threshold": 2e-4,
            "not_copied_from_single_roi": True,
            "single_roi_was": dict(SINGLE_ROI_BUDGET),
            "note": (
                "Implementation check on a few spread tiles. Dummy L0 zoom targets "
                "are not reconstruction scores. Densify steps use a 1x train view "
                "so dummy zoom loss cannot starve splitting. Not a config search "
                "on test metrics."
            ),
        }
    n = max(int(n_tiles), 1)
    steps = int(math.ceil(int(visits_per_tile) * n / (1.0 - mix)))
    densify_until = max(20, int(0.7 * steps))
    max_points = min(200_000, max(20_000, n * 4_000))
    return {
        "mode": "joint_full",
        "n_tiles": n,
        "mix_ratio": mix,
        "visits_per_tile_expected": int(visits_per_tile),
        "visits_are_expected_not_guaranteed": True,
        "sampling": "random_with_replacement",
        "steps": steps,
        "max_points": int(max_points),
        "densify_from": 1,
        "densify_until": int(densify_until),
        "densify_interval": 10,
        "densify_grad_threshold": 2e-4,
        "not_copied_from_single_roi": True,
        "single_roi_was": dict(SINGLE_ROI_BUDGET),
        "training_volume": (
            f"{steps} steps; expected {visits_per_tile} visits per tile "
            f"({steps}*(1-{mix:g})/{n} = {visits_per_tile}) under uniform random "
            "draws. Actual min/median/max/unvisited are recorded after training."
        ),
        "note": (
            "One shared L1 over all 2x tiles, then one shared L2 over all 4x tiles. "
            "Random sampling is unchanged. Record per-tile visit stats. "
            "Do not copy the single-ROI 500 / 50k budget."
        ),
    }


def coverage_payload(
    *,
    tiles_2x: Sequence[Mapping[str, Any]],
    occupancy: Mapping[str, Any],
    train_names: Sequence[str],
    test_names: Sequence[str],
    cameras_extent: float,
    min_world_dist: float,
    eval_windows: Sequence[Mapping[str, Any]] | None = None,
    occupancy_4x: Mapping[str, Any] | None = None,
    image_coverage_2x: Mapping[str, Any] | None = None,
    image_coverage_4x: Mapping[str, Any] | None = None,
    view_supervision: Mapping[str, Any] | None = None,
    width: int = 2048,
    height: int = 2048,
) -> dict[str, Any]:
    tiles = [dict(item) for item in tiles_2x]
    tiles_4x = nested_4x_tiles(tiles, width=width, height=height)
    views_with = sorted({item["view_name"] for item in tiles})
    views_without = sorted(set(train_names) - set(views_with))
    probe_ids = farthest_probe_ids(tiles)
    n = len(tiles)
    payload = {
        "scene": PROTOCOL["scene"],
        "split": {
            "train": list(train_names),
            "test": list(test_names),
            "test_images_excluded_from": list(PROTOCOL["test_images_excluded_from"]),
        },
        "grid": {
            "zoom": 2.0,
            "overlap": GRID_OVERLAP,
            "min_mask_frac": MIN_MASK_FRAC,
            "centers": list(overlapping_centers(2.0)),
            "same_center_4x": True,
            "nested_4x_is_not_automatic_scene_cover": True,
        },
        "tiles_2x": tiles,
        "tiles_4x": tiles_4x,
        "n_tiles_2x": n,
        "n_tiles_4x": len(tiles_4x),
        "stage_views_l1": [item["view_name"] for item in primary_tiles_by_view(tiles)],
        "views_with_tiles": views_with,
        "views_without_tiles": views_without,
        "occupancy_2x_centers": dict(occupancy),
        "occupancy": dict(occupancy),
        "occupancy_4x_footprints": dict(occupancy_4x or {}),
        "image_coverage_2x": dict(image_coverage_2x or {}),
        "image_coverage_4x": dict(image_coverage_4x or {}),
        "view_supervision": dict(view_supervision or {}),
        "cameras_extent": float(cameras_extent),
        "min_world_dist": float(min_world_dist),
        "min_world_dist_frac": MIN_WORLD_DIST_FRAC,
        "probe_tiles": probe_ids,
        "eval_windows": list(eval_windows or locked_eval_windows(test_names, width=width, height=height)),
        "overlap_windows": dict(OVERLAP_WINDOWS),
        "mask_policy": dict(MASK_POLICY),
        "lod_gain_baseline": dict(LOD_GAIN_BASELINE),
        "budget_full": joint_budget(n, probe=False),
        "budget_probe": joint_budget(n, probe=True, n_probe_tiles=len(probe_ids) or PROBE_TILES),
        "do_not_claim": [
            "high_frequency_detail_in_2048_is_real",
            "success_rate_over_overlapping_windows",
            "upsampled_gt_is_high_res_reference",
            "2x_center_occupancy_implies_4x_scene_cover",
            "spatial_dedup_equals_view_coverage",
        ],
        "stitch_independent_roi_models": False,
        "geometry_mainline_only": True,
    }
    return payload
