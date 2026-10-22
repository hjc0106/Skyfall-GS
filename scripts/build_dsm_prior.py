#!/usr/bin/env python3
"""Build a Skyfall-world DSM prior from DFC2019 + initial SfM points3D.

XY comes from scene.loc ECEF origin (no free 2D fit). Δz is the MAD-filtered
median residual between points3D and the DSM top surface — never from a
trained Gaussian reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dsm_georef import (  # noqa: E402
    colorize,
    load_dsm,
    load_enu_origin,
    write_xyz_rgb_ply,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = "JAX_068"


def load_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            xyz.append([float(elems[1]), float(elems[2]), float(elems[3])])
            rgb.append([int(elems[4]), int(elems[5]), int(elems[6])])
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.float64) / 255.0


def dsm_rowcol(east: np.ndarray, north: np.ndarray, dsm: dict) -> tuple[np.ndarray, np.ndarray]:
    col = (east - dsm["west"]) / dsm["gsd"] - 0.5
    row = (dsm["north"] - north) / dsm["gsd"] - 0.5
    return row, col


def bilinear_sample(grid: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    h, w = grid.shape[:2]
    r0 = np.floor(row).astype(np.int32)
    c0 = np.floor(col).astype(np.int32)
    r1 = r0 + 1
    c1 = c0 + 1
    inside = (r0 >= 0) & (c0 >= 0) & (r1 < h) & (c1 < w)
    dr = np.clip(row - r0, 0.0, 1.0)
    dc = np.clip(col - c0, 0.0, 1.0)
    out_shape = (row.shape[0],) + grid.shape[2:]
    out = np.full(out_shape, np.nan, dtype=np.float64)
    if not np.any(inside):
        return out
    r0i, c0i, r1i, c1i = r0[inside], c0[inside], r1[inside], c1[inside]
    dri, dci = dr[inside], dc[inside]
    g00 = grid[r0i, c0i]
    g01 = grid[r0i, c1i]
    g10 = grid[r1i, c0i]
    g11 = grid[r1i, c1i]
    if grid.ndim == 2:
        dri = dri
        dci = dci
        out[inside] = (
            g00 * (1 - dri) * (1 - dci)
            + g01 * (1 - dri) * dci
            + g10 * dri * (1 - dci)
            + g11 * dri * dci
        )
    else:
        dri = dri[:, None]
        dci = dci[:, None]
        out[inside] = (
            g00 * (1 - dri) * (1 - dci)
            + g01 * (1 - dri) * dci
            + g10 * dri * (1 - dci)
            + g11 * dri * dci
        )
    return out


def estimate_dz(xyz: np.ndarray, origin: dict, dsm: dict, mad_k: float = 3.5) -> dict:
    east = origin["utm_easting"] + xyz[:, 0]
    north = origin["utm_northing"] + xyz[:, 1]
    height_raw = origin["alt"] + xyz[:, 2]
    row, col = dsm_rowcol(east, north, dsm)
    h_dsm = bilinear_sample(dsm["height"], row, col)
    inside = np.isfinite(h_dsm)
    residual = height_raw[inside] - h_dsm[inside]
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        inliers = np.ones_like(residual, dtype=bool)
    else:
        inliers = np.abs(residual - med) <= mad_k * scale
    dz = -float(np.median(residual[inliers]))
    return {
        "dz": dz,
        "n_inside": int(inside.sum()),
        "n_inlier": int(inliers.sum()),
        "median_residual_before": med,
        "mad": mad,
        "median_residual_after": float(np.median(residual[inliers] + dz)),
        "mae_inlier_after": float(np.median(np.abs(residual[inliers] + dz))),
    }


def finite_difference_normals(height: np.ndarray, gsd: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(height, gsd, gsd)
    # row increases south, so world +y (north) is -row.
    n = np.stack((-dz_dx, dz_dy, np.ones_like(height)), axis=-1)
    n /= np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8, None)
    return n.astype(np.float32)


def build_confidence(height: np.ndarray, gsd: float, fade_m: float = 3.0, jump_m: float = 1.5) -> np.ndarray:
    valid = np.isfinite(height)
    conf = np.zeros(height.shape, dtype=np.float32)
    conf[valid] = 1.0
    mag = np.hypot(*np.gradient(np.nan_to_num(height, nan=np.nanmedian(height))))
    jump = mag > jump_m
    conf[jump & valid] = np.minimum(conf[jump & valid], 0.5)

    dist = distance_to_invalid(valid).astype(np.float32) * gsd
    fade_px_band = fade_m
    edge = valid & (dist <= fade_px_band)
    conf[edge] *= np.clip(dist[edge] / fade_px_band, 0.0, 1.0)
    conf[~valid] = 0.0
    return conf


def distance_to_invalid(valid: np.ndarray) -> np.ndarray:
    """Chebyshev distance (pixels) from each valid cell to the nearest invalid/border cell."""
    h, w = valid.shape
    inf = h + w + 1
    dist = np.full((h, w), inf, dtype=np.int32)
    dist[valid] = inf
    # seed invalid and image border
    dist[~valid] = 0
    dist[0, :] = 0
    dist[-1, :] = 0
    dist[:, 0] = 0
    dist[:, -1] = 0
    for _ in range(2):
        for r in range(h):
            for c in range(w):
                v = dist[r, c]
                if r:
                    v = min(v, dist[r - 1, c] + 1)
                if c:
                    v = min(v, dist[r, c - 1] + 1)
                dist[r, c] = v
        for r in range(h - 1, -1, -1):
            for c in range(w - 1, -1, -1):
                v = dist[r, c]
                if r + 1 < h:
                    v = min(v, dist[r + 1, c] + 1)
                if c + 1 < w:
                    v = min(v, dist[r, c + 1] + 1)
                dist[r, c] = v
    dist[~valid] = 0
    return dist


def make_holdout(h: int, w: int, row0: int, col0: int, size: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r1 = min(h, row0 + size)
    c1 = min(w, col0 + size)
    mask[row0:r1, col0:c1] = True
    return mask


def coarse_edges(h: int, w: int, step: int, valid: np.ndarray) -> tuple[np.ndarray, int, int]:
    ch = (h + step - 1) // step
    cw = (w + step - 1) // step
    pairs = []
    for cr in range(ch):
        for cc in range(cw):
            a = cr * cw + cc
            r, c = cr * step, cc * step
            if r >= h or c >= w or not valid[r, c]:
                continue
            if cc + 1 < cw:
                r2, c2 = r, (cc + 1) * step
                if c2 < w and valid[r2, c2]:
                    pairs.append((a, cr * cw + (cc + 1)))
            if cr + 1 < ch:
                r2, c2 = (cr + 1) * step, c
                if r2 < h and valid[r2, c2]:
                    pairs.append((a, (cr + 1) * cw + cc))
    edges = np.asarray(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), dtype=np.int32)
    return edges, ch, cw


def world_grid(dsm: dict, origin: dict, dz: float) -> dict:
    size, gsd = dsm["size"], dsm["gsd"]
    yy, xx = np.mgrid[0:size, 0:size]
    east = dsm["west"] + (xx + 0.5) * gsd
    north = dsm["north"] - (yy + 0.5) * gsd
    x = east - origin["utm_easting"]
    y = north - origin["utm_northing"]
    z = dsm["height"] - origin["alt"] - dz
    x0 = float(dsm["west"] + 0.5 * gsd - origin["utm_easting"])
    y0 = float(dsm["north"] - 0.5 * gsd - origin["utm_northing"])
    return {"x": x.astype(np.float32), "y": y.astype(np.float32), "z": z.astype(np.float32), "x0": x0, "y0": y0}


def project_points(xyz: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    cam = (w2c @ homo.T).T
    z = cam[:, 2]
    u = frame["fl_x"] * cam[:, 0] / np.clip(z, 1e-6, None) + frame["cx"]
    v = frame["fl_y"] * cam[:, 1] / np.clip(z, 1e-6, None) + frame["cy"]
    return u, v, z


def save_alignment_preview(
    out_dir: Path,
    dsm: dict,
    origin: dict,
    dz: float,
    xyz: np.ndarray,
    rgb: np.ndarray,
    height_world: np.ndarray,
    scene_dir: Path,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    east = origin["utm_easting"] + xyz[:, 0]
    north = origin["utm_northing"] + xyz[:, 1]
    z_reg = origin["alt"] + xyz[:, 2] + dz
    row, col = dsm_rowcol(east, north, dsm)
    ri = np.rint(row).astype(np.int32)
    ci = np.rint(col).astype(np.int32)
    h, w = dsm["size"], dsm["size"]
    inside = (ri >= 0) & (ci >= 0) & (ri < h) & (ci < w)
    pred = np.full((h, w), np.nan)
    gs_rgb = np.zeros((h, w, 3))
    order = np.argsort(z_reg[inside])
    pred[ri[inside][order], ci[inside][order]] = z_reg[inside][order]
    gs_rgb[ri[inside][order], ci[inside][order]] = rgb[inside][order]
    error = pred - dsm["height"]
    valid = np.isfinite(pred)

    vmin, vmax = float(np.nanmin(dsm["height"])), float(np.nanmax(dsm["height"]))
    dsm_rgb = colorize(dsm["height"], "terrain", vmin, vmax)
    pred_rgb = colorize(np.where(valid, pred, np.nan), "terrain", vmin, vmax)
    blend = dsm_rgb.copy()
    blend[valid] = 0.45 * dsm_rgb[valid] + 0.55 * gs_rgb[valid]
    err_rgb = colorize(error, "coolwarm", -8.0, 8.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, img, title in [
        (axes[0, 0], dsm_rgb, "GT DSM"),
        (axes[0, 1], pred_rgb, "SfM points3D (Z-registered)"),
        (axes[1, 0], np.clip(blend, 0, 1), "DSM + SfM color"),
        (axes[1, 1], err_rgb, "SfM − DSM after Δz (m)"),
    ]:
        ax.imshow(img, origin="upper")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_dir / "sfm_on_dsm.png", dpi=140)
    plt.close(fig)

    if np.any(valid):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.clip(error[valid], -20, 20), bins=80, color="#4c78a8")
        ax.set_xlabel("SfM − DSM (m)")
        ax.set_xlim(-20, 20)
        ax.set_title("points3D residual after Δz")
        fig.tight_layout()
        fig.savefig(out_dir / "residual_hist.png", dpi=120)
        plt.close(fig)

    overlays = save_image_overlays(out_dir / "image_overlays", scene_dir, height_world, dsm, origin, xyz, rgb)
    return ["sfm_on_dsm.png", "residual_hist.png"] + overlays


def save_image_overlays(
    out_dir: Path,
    scene_dir: Path,
    height_world: np.ndarray,
    dsm: dict,
    origin: dict,
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_images: int = 6,
    step: int = 4,
) -> list[str]:
    transforms_path = scene_dir / "transforms_train.json"
    frames = json.loads(transforms_path.read_text())["frames"]
    size, gsd = dsm["size"], dsm["gsd"]
    yy, xx = np.mgrid[0:size:step, 0:size:step]
    east = dsm["west"] + (xx + 0.5) * gsd
    north = dsm["north"] - (yy + 0.5) * gsd
    dsm_xyz = np.stack(
        [
            east - origin["utm_easting"],
            north - origin["utm_northing"],
            height_world[::step, ::step],
        ],
        axis=-1,
    ).reshape(-1, 3)
    dsm_h = dsm["height"][::step, ::step].reshape(-1)
    keep = np.isfinite(dsm_xyz[:, 2])
    dsm_xyz, dsm_h = dsm_xyz[keep], dsm_h[keep]
    h_min, h_max = float(dsm_h.min()), float(dsm_h.max())
    dsm_color = colorize(dsm_h, "turbo", h_min, h_max)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for frame in frames[:max_images]:
        image_path = scene_dir / frame["file_path"]
        if not image_path.exists():
            continue
        img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        hh, ww = img.shape[:2]
        canvas = img.copy()
        u, v, z = project_points(dsm_xyz, frame)
        vis = (z > 0) & (u >= 0) & (v >= 0) & (u < ww) & (v < hh)
        ui = np.clip(np.rint(u[vis]).astype(np.int32), 1, ww - 2)
        vi = np.clip(np.rint(v[vis]).astype(np.int32), 1, hh - 2)
        color = dsm_color[vis]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                canvas[vi + dy, ui + dx] = 0.15 * canvas[vi + dy, ui + dx] + 0.85 * color

        su, sv, sz = project_points(xyz, frame)
        svis = (sz > 0) & (su >= 0) & (sv >= 0) & (su < ww) & (sv < hh)
        sui = np.clip(np.rint(su[svis]).astype(np.int32), 0, ww - 1)
        svi = np.clip(np.rint(sv[svis]).astype(np.int32), 0, hh - 1)
        canvas[svi, sui] = 0.4 * canvas[svi, sui] + 0.6 * rgb[svis]

        name = Path(frame["file_path"]).stem + "_dsm_overlay.png"
        Image.fromarray(np.clip(canvas * 255, 0, 255).astype(np.uint8)).save(out_dir / name)
        written.append(f"image_overlays/{name}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-previews", action="store_true")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--dsm", type=Path, default=None)
    parser.add_argument("--dsm-txt", type=Path, default=None)
    parser.add_argument("--scene-loc", type=Path, default=None)
    parser.add_argument("--points3d", type=Path, default=None)
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-step", type=int, default=2)
    parser.add_argument("--holdout-row", type=int, default=320)
    parser.add_argument("--holdout-col", type=int, default=320)
    parser.add_argument("--holdout-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_paths(args):
    for key in ('dsm', 'scene_loc', 'scene_dir', 'output'):
        if getattr(args, key) is None:
            raise ValueError('--' + key.replace('_', '-') + ' is required')
    args.dsm_txt = args.dsm_txt or args.dsm.with_suffix('.txt')
    args.points3d = args.points3d or args.scene_dir / 'points3D.txt'
    if args.sample_step <= 0:
        raise ValueError('sample-step must be positive')
    if args.output.resolve() == args.scene_dir.resolve() or args.scene_dir.resolve() in args.output.resolve().parents:
        raise ValueError('Output must be outside source dataset')
    contents = json.loads((args.scene_dir / 'transforms_train.json').read_text())
    if 'R' in contents:
        raise ValueError('Transform this prior into the normalized training frame before use')
    return args


def main() -> None:
    args = resolve_paths(parse_args())
    origin = load_enu_origin(args.scene_loc)
    dsm = load_dsm(args.dsm, args.dsm_txt)
    xyz, rgb = load_points3d(args.points3d)
    stats = estimate_dz(xyz, origin, dsm)
    dz = stats["dz"]

    world = world_grid(dsm, origin, dz)
    height_world = world["z"]
    valid = np.isfinite(dsm["height"])
    normals = finite_difference_normals(dsm["height"], dsm["gsd"])
    confidence = build_confidence(dsm["height"], dsm["gsd"])
    holdout = make_holdout(dsm["size"], dsm["size"], args.holdout_row, args.holdout_col, args.holdout_size)
    confidence = confidence.copy()
    confidence[holdout] = 0.0
    edges, ch, cw = coarse_edges(dsm["size"], dsm["size"], args.sample_step, valid & ~holdout & (confidence > 0))

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "dsm_prior.npz",
        height_world=height_world,
        height_utm=dsm["height"].astype(np.float32),
        normal=normals,
        valid=valid,
        holdout=holdout,
        confidence=confidence,
        edges=edges,
        x0=np.float32(world["x0"]),
        y0=np.float32(world["y0"]),
        gsd=np.float32(dsm["gsd"]),
        x_grid=world["x"],
        y_grid=world["y"],
        sample_step=np.int32(args.sample_step),
        coarse_h=np.int32(ch),
        coarse_w=np.int32(cw),
        utm_easting=np.float32(origin["utm_easting"]),
        utm_northing=np.float32(origin["utm_northing"]),
        alt=np.float32(origin["alt"]),
        dz=np.float32(dz),
        dsm_west=np.float32(dsm["west"]),
        dsm_south=np.float32(dsm["south"]),
        dsm_north=np.float32(dsm["north"]),
        dsm_size=np.int32(dsm["size"]),
    )

    registration = {
        "scene": args.scene,
        "enu_origin": {
            "lat": origin["lat"],
            "lon": origin["lon"],
            "alt_m": origin["alt"],
            "utm_easting": origin["utm_easting"],
            "utm_northing": origin["utm_northing"],
            "utm_zone": origin["utm_zone"],
        },
        "transform": {
            "E": "E0 + x",
            "N": "N0 + y",
            "H": "h0 + z + dz",
            "E0": origin["utm_easting"],
            "N0": origin["utm_northing"],
            "h0": origin["alt"],
            "dz": dz,
            "xy": "from scene.loc, no free fit",
            "dz_source": "median MAD-filtered residual of initial points3D vs DSM",
        },
        "dsm_bounds_utm": {
            "west": dsm["west"],
            "south": dsm["south"],
            "east": dsm["east"],
            "north": dsm["north"],
            "gsd_m": dsm["gsd"],
            "size": dsm["size"],
        },
        "holdout": {
            "row": args.holdout_row,
            "col": args.holdout_col,
            "size": args.holdout_size,
            "n_cells": int(holdout.sum()),
        },
        "residual_stats": stats,
    }
    import hashlib
    registration['schema_version'] = 1
    registration['crs'] = {'epsg': 32617, 'source': 'explicit UTM 17N interpretation of sidecar values', 'vertical_datum': 'unknown'}
    registration['source_hashes'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (args.points3d, args.dsm, args.dsm_txt, args.scene_loc)}
    registration['reader_transform'] = 'identity (source R/T absent)'
    registration['coordinate_gate'] = {'measured_track_reprojection': 'not_verified', 'training_ready': False}
    (args.output / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")

    dsm_xyz = np.stack([world["x"], world["y"], height_world], axis=-1).reshape(-1, 3)
    dsm_rgb = colorize(dsm["height"], "terrain", float(dsm["height"].min()), float(dsm["height"].max())).reshape(-1, 3)
    write_xyz_rgb_ply(args.output / "dsm_surface.ply", dsm_xyz, dsm_rgb)

    preview_files = [] if args.skip_previews else save_alignment_preview(
        args.output / "alignment_preview",
        dsm,
        origin,
        dz,
        xyz,
        rgb,
        height_world,
        args.scene_dir,
    )

    summary = {
        "scene": args.scene,
        "output": str(args.output),
        "n_points3d": int(xyz.shape[0]),
        "n_valid_dsm": int(valid.sum()),
        "n_holdout": int(holdout.sum()),
        "n_edges": int(edges.shape[0]),
        "confidence_mean": float(confidence[valid & ~holdout].mean()) if np.any(valid & ~holdout) else 0.0,
        "dz": dz,
        "residual_stats": stats,
        "previews": preview_files,
        "note": "Inspect alignment_preview/image_overlays before training. DSM dots are height-colored; SfM points use original RGB.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
