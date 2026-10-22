"""Evaluate a trained Skyfall-GS scene against the DSM prior.

Geometry metrics are computed on Gaussian centers vs the registered DSM grid,
including a held-out spatial block that was excluded from init and DSM losses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.dsm_prior import load_npz


def load_ply_xyz_opacity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply.elements[0]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)
    if "opacity" in v.data.dtype.names:
        logit = np.asarray(v["opacity"], dtype=np.float64)
        alpha = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    else:
        alpha = np.ones(xyz.shape[0], dtype=np.float64)
    return xyz, alpha


def load_ply_normals(path: Path) -> np.ndarray:
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation
    v = PlyData.read(str(path)).elements[0]
    names = v.data.dtype.names
    normal = np.zeros((len(v.data), 3), dtype=np.float64)
    normal[:, 2] = 1.0
    scale_names = [f"scale_{i}" for i in range(3)]
    rot_names = [f"rot_{i}" for i in range(4)]
    if all(name in names for name in scale_names + rot_names):
        scales = np.stack([np.asarray(v[name]) for name in scale_names], axis=1)
        quat_wxyz = np.stack([np.asarray(v[name]) for name in rot_names], axis=1)
        rotation = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()
        normal_axis = np.argmin(scales, axis=1)
        normal = rotation[np.arange(len(v.data)), :, normal_axis]
    return normal


def raster_max_z(xyz: np.ndarray, prior: dict) -> np.ndarray:
    x0, y0, gsd = float(prior["x0"]), float(prior["y0"]), float(prior["gsd"])
    h, w = prior["height_world"].shape
    col = np.floor((xyz[:, 0] - x0) / gsd).astype(np.int32)
    row = np.floor((y0 - xyz[:, 1]) / gsd).astype(np.int32)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    pred = np.full((h, w), np.nan, dtype=np.float64)
    if not np.any(inside):
        return pred
    order = np.argsort(xyz[inside, 2])
    pred[row[inside][order], col[inside][order]] = xyz[inside, 2][order]
    return pred


def height_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    sel = mask & np.isfinite(pred) & np.isfinite(gt)
    if not np.any(sel):
        return {"n": 0, "mae": None, "rmse": None, "p90": None}
    err = pred[sel] - gt[sel]
    return {
        "n": int(sel.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "p90": float(np.percentile(np.abs(err), 90)),
        "median": float(np.median(err)),
    }


def relative_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    diffs = []
    h, w = gt.shape
    for r in range(h - 1):
        m = mask[r] & mask[r + 1] & np.isfinite(pred[r]) & np.isfinite(pred[r + 1])
        if np.any(m):
            diffs.append((pred[r, m] - pred[r + 1, m]) - (gt[r, m] - gt[r + 1, m]))
    for c in range(w - 1):
        m = mask[:, c] & mask[:, c + 1] & np.isfinite(pred[:, c]) & np.isfinite(pred[:, c + 1])
        if np.any(m):
            diffs.append((pred[m, c] - pred[m, c + 1]) - (gt[m, c] - gt[m, c + 1]))
    if not diffs:
        return {"n": 0, "mae": None}
    d = np.concatenate(diffs)
    return {"n": int(d.size), "mae": float(np.mean(np.abs(d))), "p90": float(np.percentile(np.abs(d), 90))}


def normal_consistency(xyz: np.ndarray, gs_normal: np.ndarray, alpha: np.ndarray, prior: dict, surface_t: float = 3.0, min_alpha: float = 0.05) -> dict:
    height = prior["height_world"]
    normal = prior["normal"]
    x0, y0, gsd = float(prior["x0"]), float(prior["y0"]), float(prior["gsd"])
    h, w = height.shape
    col = np.clip(np.rint((xyz[:, 0] - x0) / gsd).astype(np.int32), 0, w - 1)
    row = np.clip(np.rint((y0 - xyz[:, 1]) / gsd).astype(np.int32), 0, h - 1)
    inside = (xyz[:, 0] >= x0) & (xyz[:, 0] <= x0 + (w - 1) * gsd)
    inside &= (xyz[:, 1] <= y0) & (xyz[:, 1] >= y0 - (h - 1) * gsd)
    inside &= prior["valid"][row, col] & (~prior["holdout"][row, col])
    dz = np.abs(xyz[:, 2] - height[row, col])
    on = inside & (dz <= surface_t) & (alpha >= min_alpha)
    if not np.any(on):
        return {"n": 0, "mean_abs_dot": None}
    n_dsm = normal[row[on], col[on]]
    dots = np.abs(np.sum(n_dsm * gs_normal[on], axis=1))
    return {"n": int(on.sum()), "mean_abs_dot": float(dots.mean()), "mean_1_minus_abs_dot": float((1 - dots).mean())}


def edge_preservation(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, jump: float = 1.5) -> dict:
    gy, gx = np.gradient(gt)
    mag = np.hypot(gx, gy)
    edge = mask & (mag > jump) & np.isfinite(pred)
    if not np.any(edge):
        return {"n": 0, "mae": None}
    err = np.abs(pred[edge] - gt[edge])
    return {"n": int(edge.sum()), "mae": float(err.mean()), "p90": float(np.percentile(err, 90))}


def _anchor_ids(xyz: np.ndarray, prior: dict) -> np.ndarray:
    x0, y0, gsd = float(prior["x0"]), float(prior["y0"]), float(prior["gsd"])
    h, w = prior["height_world"].shape
    step = int(prior["sample_step"])
    ch, cw = int(prior["coarse_h"]), int(prior["coarse_w"])
    col = np.floor((xyz[:, 0] - x0) / gsd).astype(np.int32)
    row = np.floor((y0 - xyz[:, 1]) / gsd).astype(np.int32)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    aid = np.full((xyz.shape[0],), -1, dtype=np.int32)
    cr = np.clip(row // step, 0, ch - 1)
    cc = np.clip(col // step, 0, cw - 1)
    aid[inside] = cr[inside] * cw + cc[inside]
    return aid


def _anchor_region_masks(prior: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = int(prior["sample_step"])
    ch, cw = int(prior["coarse_h"]), int(prior["coarse_w"])
    H, W = prior["height_world"].shape
    holdout = prior["holdout"].astype(bool)
    valid = prior["valid"].astype(bool)
    gt = prior["height_world"].astype(np.float64)
    n_a = ch * cw
    train = np.zeros(n_a, dtype=bool)
    hold = np.zeros(n_a, dtype=bool)
    height = np.zeros(n_a, dtype=np.float64)
    cr = np.arange(ch)
    cc = np.arange(cw)
    rr = np.minimum(cr[:, None] * step, H - 1)
    ck = np.minimum(cc[None, :] * step, W - 1)
    idx = cr[:, None] * cw + cc[None, :]
    height[idx] = gt[rr, ck]
    v = valid[rr, ck]
    ho = holdout[rr, ck]
    train[idx] = v & ~ho
    hold[idx] = v & ho
    return train, hold, height


def _grid_neighbor_edges(ch: int, cw: int, mask: np.ndarray) -> np.ndarray:
    idx = np.arange(ch * cw, dtype=np.int64).reshape(ch, cw)
    m = mask.reshape(ch, cw)
    parts = []
    horiz = m[:, :-1] & m[:, 1:]
    if np.any(horiz):
        parts.append(np.stack([idx[:, :-1][horiz], idx[:, 1:][horiz]], axis=1))
    vert = m[:-1, :] & m[1:, :]
    if np.any(vert):
        parts.append(np.stack([idx[:-1, :][vert], idx[1:, :][vert]], axis=1))
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(parts, axis=0)


def anchor_mean_relative(
    xyz: np.ndarray,
    alpha: np.ndarray,
    prior: dict,
    min_alpha: float = 0.05,
) -> dict:
    """Opacity-weighted mean height per coarse anchor, then edge relative MAE.

    Matches the training relative loss target (anchor-mean z), not the
    evaluation top-surface (max-z) raster. Train edges are the prior graph
    used in the loss. Holdout edges are 4-neighbor pairs inside the white
    box only, so the metric can be reported without leaking the loss graph
    into the holdout.
    """
    ch, cw = int(prior["coarse_h"]), int(prior["coarse_w"])
    n_a = ch * cw
    train_edges = np.asarray(prior["edges"], dtype=np.int64)
    sel = alpha >= min_alpha
    aid = _anchor_ids(xyz[sel], prior)
    z = xyz[sel, 2]
    w = alpha[sel]
    ok = aid >= 0
    sum_az = np.zeros(n_a, dtype=np.float64)
    sum_a = np.zeros(n_a, dtype=np.float64)
    np.add.at(sum_az, aid[ok], w[ok] * z[ok])
    np.add.at(sum_a, aid[ok], w[ok])
    occupied = sum_a > 1e-6
    mean_z = np.full(n_a, np.nan, dtype=np.float64)
    mean_z[occupied] = sum_az[occupied] / sum_a[occupied]
    train_m, hold_m, height = _anchor_region_masks(prior)
    hold_edges = _grid_neighbor_edges(ch, cw, hold_m)

    def _on_edges(edges: np.ndarray, region: np.ndarray) -> dict:
        n_occ = int((occupied & region).sum())
        if edges.size == 0:
            return {"n": 0, "n_occupied_anchors": n_occ, "mae": None, "p90": None}
        a, b = edges[:, 0], edges[:, 1]
        both = occupied[a] & occupied[b] & region[a] & region[b]
        if not np.any(both):
            return {"n": 0, "n_occupied_anchors": n_occ, "mae": None, "p90": None}
        err = np.abs((mean_z[a[both]] - mean_z[b[both]]) - (height[a[both]] - height[b[both]]))
        return {
            "n": int(err.size),
            "n_occupied_anchors": n_occ,
            "mae": float(err.mean()),
            "p90": float(np.percentile(err, 90)),
        }

    return {
        "train": _on_edges(train_edges, train_m),
        "holdout": _on_edges(hold_edges, hold_m),
    }


def floater_ratio(xyz: np.ndarray, alpha: np.ndarray, prior: dict, thresh: float = 3.0, min_alpha: float = 0.05) -> dict:
    height = prior["height_world"]
    x0, y0, gsd = float(prior["x0"]), float(prior["y0"]), float(prior["gsd"])
    h, w = height.shape
    col = np.floor((xyz[:, 0] - x0) / gsd).astype(np.int32)
    row = np.floor((y0 - xyz[:, 1]) / gsd).astype(np.int32)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h) & prior["valid"][np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)]
    opaque = alpha >= min_alpha
    in_cov = inside & opaque & (~prior["holdout"][np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)])
    n_in = int(in_cov.sum())
    if n_in == 0:
        return {"n_in_dsm": 0, "n_floaters": 0, "ratio": None}
    dz = xyz[:, 2] - height[np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)]
    n_f = int((in_cov & (dz > thresh)).sum())
    return {"n_in_dsm": n_in, "n_floaters": n_f, "ratio": n_f / n_in}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-m", "--model_path", required=True)
    p.add_argument("--dsm_prior", required=True)
    p.add_argument("--iteration", type=int, default=10000)
    p.add_argument("--output", default="")
    p.add_argument("--min-opacity", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = Path(args.model_path)
    ply = model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(ply)
    prior = load_npz(args.dsm_prior)
    xyz, alpha = load_ply_xyz_opacity(ply)
    gs_normal = load_ply_normals(ply)
    pred = raster_max_z(xyz[alpha >= args.min_opacity], prior)
    anchor_rel = anchor_mean_relative(xyz, alpha, prior, min_alpha=args.min_opacity)
    gt = prior["height_world"].astype(np.float64)
    valid = prior["valid"].astype(bool)
    holdout = prior["holdout"].astype(bool)
    train_mask = valid & ~holdout
    rel_train = relative_error(pred, gt, train_mask)
    rel_hold = relative_error(pred, gt, holdout & valid)
    edge_train = edge_preservation(pred, gt, train_mask)
    edge_hold = edge_preservation(pred, gt, holdout & valid)
    hold_valid_n = int((holdout & valid).sum())
    hold_filled = int((holdout & valid & np.isfinite(pred)).sum())
    report = {
        "protocol": "controlled_same_dsm",
        "surface_estimator": "opacity_filtered_gaussian_centers_max_z",
        "model_path": str(model),
        "iteration": args.iteration,
        "n_gaussians": int(xyz.shape[0]),
        "train_region": height_metrics(pred, gt, train_mask),
        "holdout_region": height_metrics(pred, gt, holdout & valid),
        "full_dsm": height_metrics(pred, gt, valid),
        "top_surface_relative_train": rel_train,
        "top_surface_relative_holdout": rel_hold,
        "relative_train": rel_train,
        "relative_holdout": rel_hold,
        "edge_train": edge_train,
        "edge_holdout": edge_hold,
        "top_surface_edge_train": edge_train,
        "top_surface_edge_holdout": edge_hold,
        "anchor_mean_relative_train": anchor_rel["train"],
        "anchor_mean_relative_holdout": anchor_rel["holdout"],
        "floaters": floater_ratio(xyz, alpha, prior),
        "normal": normal_consistency(xyz, gs_normal, alpha, prior, min_alpha=args.min_opacity),
        "coverage": {
            "valid_dsm_cells": int(valid.sum()),
            "full_fill_rate": float((valid & np.isfinite(pred)).sum() / max(valid.sum(), 1)),
            "train_cells_filled": int((train_mask & np.isfinite(pred)).sum()),
            "holdout_cells_filled": hold_filled,
            "holdout_valid": hold_valid_n,
            "holdout_fill_rate": float(hold_filled / max(hold_valid_n, 1)),
        },
    }
    out = Path(args.output) if args.output else model / f"dsm_eval_iter_{args.iteration}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    pred_path = out.with_name(f"dsm_pred_iter_{args.iteration}.npz")
    np.savez_compressed(pred_path, pred=pred.astype(np.float32))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    print(f"wrote {pred_path}")


if __name__ == "__main__":
    main()
