"""DSM prior: mixed initialization, bilinear sampling, and soft geometry losses."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_rotation


def load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files}
    required = {'height_world', 'normal', 'confidence', 'valid', 'holdout', 'edges',
                'x0', 'y0', 'gsd', 'sample_step', 'coarse_h', 'coarse_w'}
    if not required.issubset(raw):
        raise ValueError(f'Missing DSM fields: {sorted(required - raw.keys())}')
    shape = raw['height_world'].shape
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError('DSM height must be an HxW grid of at least 2x2')
    for name in ('confidence', 'valid', 'holdout'):
        if raw[name].shape != shape:
            raise ValueError(f'DSM {name} shape mismatch')
    if raw['normal'].shape != shape + (3,):
        raise ValueError('DSM normal shape mismatch')
    valid = raw['valid'].astype(bool)
    if not valid.any() or not np.isfinite(raw['height_world'][valid]).all():
        raise ValueError('DSM must have finite valid heights')
    if not np.isfinite(raw['normal'][valid]).all() or not np.isfinite(raw['confidence']).all():
        raise ValueError('DSM normals/confidence must be finite')
    if np.any(raw['confidence'] < 0):
        raise ValueError('DSM confidence must be nonnegative')
    for name in ('gsd', 'sample_step', 'coarse_h', 'coarse_w'):
        if not np.isfinite(raw[name]) or float(raw[name]) <= 0:
            raise ValueError(f'DSM {name} must be positive')
    # Grid sampling must not propagate NaNs from invalid cells.
    raw['height_world'] = np.where(valid, raw['height_world'], 0)
    raw['normal'] = np.where(valid[..., None], raw['normal'], [0, 0, 1])
    raw['confidence'] = np.where(valid & ~raw['holdout'].astype(bool), raw['confidence'], 0)
    return raw


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array(
            [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s],
            dtype=np.float32,
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s], dtype=np.float32)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s], dtype=np.float32)
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s], dtype=np.float32)


def quat_from_normal(n: np.ndarray) -> np.ndarray:
    n = n / (np.linalg.norm(n) + 1e-8)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(helper, n)
    t1 = t1 / (np.linalg.norm(t1) + 1e-8)
    t2 = np.cross(n, t1)
    return _rot_to_quat(np.stack([t1, t2, n], axis=1))


def _world_to_grid(x: torch.Tensor, y: torch.Tensor, prior: "DSMPrior") -> tuple[torch.Tensor, torch.Tensor]:
    col = (x - prior.x0) / prior.gsd
    row = (prior.y0 - y) / prior.gsd
    return row, col


class DSMPrior:
    def __init__(self, packed: dict):
        self.height = packed["height"]  # (1,1,H,W)
        self.normal = packed["normal"]  # (1,3,H,W)
        self.confidence = packed["confidence"]
        self.valid = packed["valid"]
        self.holdout = packed["holdout"]
        self.edges = packed["edges"]
        self.anchor_h = packed["anchor_h"]
        self.anchor_conf = packed["anchor_conf"]
        self.x0 = packed["x0"]
        self.y0 = packed["y0"]
        self.gsd = packed["gsd"]
        self.H = packed["H"]
        self.W = packed["W"]
        self.sample_step = packed["sample_step"]
        self.coarse_h = packed["coarse_h"]
        self.coarse_w = packed["coarse_w"]
        self.n_anchor = self.coarse_h * self.coarse_w

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DSMPrior":
        raw = load_npz(path)
        height = torch.from_numpy(raw["height_world"]).float()
        normal = torch.from_numpy(raw["normal"]).float().permute(2, 0, 1)
        conf = torch.from_numpy(raw["confidence"]).float()
        valid = torch.from_numpy(raw["valid"].astype(np.bool_))
        holdout = torch.from_numpy(raw["holdout"].astype(np.bool_))
        edges = torch.from_numpy(raw["edges"]).long()
        H, W = int(height.shape[0]), int(height.shape[1])
        step = int(raw["sample_step"])
        ch, cw = int(raw["coarse_h"]), int(raw["coarse_w"])
        anchor_h = torch.zeros(ch * cw)
        anchor_c = torch.zeros(ch * cw)
        for cr in range(ch):
            for cc in range(cw):
                r, c = min(cr * step, H - 1), min(cc * step, W - 1)
                anchor_h[cr * cw + cc] = height[r, c]
                anchor_c[cr * cw + cc] = conf[r, c]
        packed = {
            "height": height[None, None],
            "normal": normal[None],
            "confidence": conf[None, None],
            "valid": valid,
            "holdout": holdout,
            "edges": edges,
            "anchor_h": anchor_h,
            "anchor_conf": anchor_c,
            "x0": float(raw["x0"]),
            "y0": float(raw["y0"]),
            "gsd": float(raw["gsd"]),
            "H": H,
            "W": W,
            "sample_step": step,
            "coarse_h": ch,
            "coarse_w": cw,
        }
        obj = cls(packed)
        return obj.to(device)

    def to(self, device: str | torch.device) -> "DSMPrior":
        self.height = self.height.to(device)
        self.normal = self.normal.to(device)
        self.confidence = self.confidence.to(device)
        self.valid = self.valid.to(device)
        self.holdout = self.holdout.to(device)
        self.edges = self.edges.to(device)
        self.anchor_h = self.anchor_h.to(device)
        self.anchor_conf = self.anchor_conf.to(device)
        return self

    def cuda(self) -> "DSMPrior":
        return self.to("cuda")

    def sample(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bilinear sample height, normal, confidence at world (x,y). xy: (N,2)."""
        row, col = _world_to_grid(xy[:, 0], xy[:, 1], self)
        nx = 2.0 * col / max(self.W - 1, 1) - 1.0
        ny = 2.0 * row / max(self.H - 1, 1) - 1.0
        grid = torch.stack([nx, ny], dim=-1).view(1, 1, -1, 2)
        h = F.grid_sample(self.height, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        n = F.grid_sample(self.normal, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        c = F.grid_sample(self.confidence, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        h = h.view(-1)
        n = n.view(3, -1).transpose(0, 1)
        n = F.normalize(n, dim=-1)
        c = c.view(-1)
        return h, n, c


def try_load_dsm_prior(dataset) -> Optional[DSMPrior]:
    path = getattr(dataset, "dsm_prior", "") or ""
    if not path:
        return None
    if not Path(path).exists():
        raise FileNotFoundError(f"DSM prior not found: {path}")
    return DSMPrior.load(path, device="cuda")


def dsm_loss_scale(iteration: int, opt, is_idu: bool = False) -> float:
    warmup = int(getattr(opt, "dsm_warmup_iters", 500))
    hold = int(getattr(opt, "dsm_hold_iters", 5000))
    decay = int(getattr(opt, "dsm_decay_iters", 21000))
    late = float(getattr(opt, "dsm_late_scale", 0.2))
    if iteration <= 0:
        s = 0.0
    elif iteration < warmup:
        s = iteration / max(warmup, 1)
    elif iteration < hold:
        s = 1.0
    elif iteration < decay:
        t = (iteration - hold) / max(decay - hold, 1)
        s = 1.0 + t * (late - 1.0)
    else:
        s = late
    if is_idu:
        s *= float(getattr(opt, "dsm_idu_scale", 0.2))
    return s


def _stratified_sample(rows: np.ndarray, cols: np.ndarray, h: int, w: int, n: int, rng: np.random.Generator) -> np.ndarray:
    n = min(n, len(rows))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    n_br = max(1, int(round(np.sqrt(n * h / max(w, 1)))))
    n_bc = max(1, int(round(n / n_br)))
    bins: dict[tuple[int, int], list[int]] = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        br = min(n_br - 1, int(r * n_br / max(h, 1)))
        bc = min(n_bc - 1, int(c * n_bc / max(w, 1)))
        bins.setdefault((br, bc), []).append(i)
    keys = list(bins.keys())
    rng.shuffle(keys)
    picks: list[int] = []
    k = 0
    while len(picks) < n and any(bins[key] for key in keys):
        key = keys[k % len(keys)]
        if bins[key]:
            j = int(rng.integers(len(bins[key])))
            picks.append(bins[key].pop(j))
        k += 1
    return np.asarray(picks[:n], dtype=np.int64)


def _project_color(xyz: np.ndarray, cameras: list) -> np.ndarray:
    colors = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    hits = np.zeros((xyz.shape[0],), dtype=np.int32)
    for cam in cameras[:8]:
        image = np.asarray(cam.image.convert("RGB"), dtype=np.float32) / 255.0
        hh, ww = image.shape[:2]
        xyz_cam = xyz @ cam.R + cam.T[None, :]
        z = xyz_cam[:, 2]
        tanx = np.tan(float(cam.FovX) / 2.0)
        tany = np.tan(float(cam.FovY) / 2.0)
        fx = ww / (2.0 * tanx)
        fy = hh / (2.0 * tany)
        cx = cam.cx / 2.0 * ww + ww / 2.0
        cy = cam.cy / 2.0 * hh + hh / 2.0
        u = fx * xyz_cam[:, 0] / np.clip(z, 1e-6, None) + cx
        v = fy * xyz_cam[:, 1] / np.clip(z, 1e-6, None) + cy
        vis = (z > 0) & (u >= 0) & (v >= 0) & (u < ww - 1) & (v < hh - 1)
        if cam.mask is not None:
            ui = np.clip(np.rint(u).astype(np.int32), 0, ww - 1)
            vi = np.clip(np.rint(v).astype(np.int32), 0, hh - 1)
            vis = vis & (cam.mask[vi, ui] > 0)
        ui = np.clip(np.rint(u[vis]).astype(np.int32), 0, ww - 1)
        vi = np.clip(np.rint(v[vis]).astype(np.int32), 0, hh - 1)
        colors[vis] += image[vi, ui]
        hits[vis] += 1
    ok = hits > 0
    colors[ok] /= hits[ok, None]
    colors[~ok] = 0.5
    return colors


def mix_init_with_dsm(pcd: BasicPointCloud, dsm_path: str, cameras: list, opt) -> tuple[BasicPointCloud, dict]:
    raw = load_npz(dsm_path)
    height = raw["height_world"].astype(np.float64)
    conf = raw["confidence"].astype(np.float64)
    holdout = raw["holdout"].astype(bool)
    valid = raw["valid"].astype(bool)
    normal = raw["normal"].astype(np.float64)
    x0, y0, gsd = float(raw["x0"]), float(raw["y0"]), float(raw["gsd"])
    H, W = height.shape
    step = int(raw["sample_step"])
    ch, cw = int(raw["coarse_h"]), int(raw["coarse_w"])
    surface_t = float(getattr(opt, "dsm_surface_thresh", 3.0))
    max_pts = int(getattr(opt, "dsm_max_init_points", 65536))
    tang = float(getattr(opt, "dsm_tangent_scale", 0.5))
    nscale = float(getattr(opt, "dsm_normal_scale", 0.2))
    rng = np.random.default_rng(int(getattr(opt, "dsm_seed", 0)))

    xyz = np.asarray(pcd.points, dtype=np.float64)
    rgb = np.asarray(pcd.colors, dtype=np.float64)
    col = (xyz[:, 0] - x0) / gsd
    row = (y0 - xyz[:, 1]) / gsd
    r0 = np.floor(row).astype(np.int32)
    c0 = np.floor(col).astype(np.int32)
    inside = (r0 >= 0) & (c0 >= 0) & (r0 < H - 1) & (c0 < W - 1) & valid[np.clip(r0, 0, H - 1), np.clip(c0, 0, W - 1)]
    h_s = np.full(xyz.shape[0], np.nan)
    if np.any(inside):
        # reuse nearest cell for classification
        h_s[inside] = height[r0[inside], c0[inside]]
    in_cover = inside & ~holdout[np.clip(r0, 0, H - 1), np.clip(c0, 0, W - 1)] & np.isfinite(h_s)
    dz = xyz[:, 2] - h_s
    near = in_cover & (np.abs(dz) <= surface_t)
    floaters = in_cover & (dz > surface_t)
    keep_sfm = ~near & ~floaters

    n_replace = int(near.sum())
    cand_r, cand_c = [], []
    for cr in range(ch):
        for cc in range(cw):
            r, c = cr * step, cc * step
            if r >= H or c >= W:
                continue
            if (not valid[r, c]) or holdout[r, c] or conf[r, c] <= 0:
                continue
            cand_r.append(r)
            cand_c.append(c)
    cand_r = np.asarray(cand_r, dtype=np.int32)
    cand_c = np.asarray(cand_c, dtype=np.int32)
    n_dsm = min(n_replace, max_pts, len(cand_r))
    pick = _stratified_sample(cand_r, cand_c, H, W, n_dsm, rng)
    sel_r, sel_c = cand_r[pick], cand_c[pick]
    n_dsm = len(sel_r)

    dsm_xyz = np.stack(
        [x0 + sel_c * gsd, y0 - sel_r * gsd, height[sel_r, sel_c]],
        axis=-1,
    )
    dsm_n = normal[sel_r, sel_c]
    dsm_c = conf[sel_r, sel_c]
    dsm_aid = (sel_r // step) * cw + (sel_c // step)

    if xyz[keep_sfm].shape[0] > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(xyz[keep_sfm | near])
        _, nn = tree.query(dsm_xyz, k=1)
        src_rgb = rgb[keep_sfm | near][nn]
    else:
        src_rgb = np.full((n_dsm, 3), 0.5, dtype=np.float64)
    missing = np.linalg.norm(src_rgb, axis=1) < 1e-6
    if np.any(missing) and cameras:
        src_rgb[missing] = _project_color(dsm_xyz[missing], cameras)

    sfm_xyz = xyz[keep_sfm]
    sfm_rgb = rgb[keep_sfm]
    out_xyz = np.concatenate([sfm_xyz, dsm_xyz], axis=0)
    out_rgb = np.concatenate([sfm_rgb, src_rgb], axis=0)
    out_n = np.zeros_like(out_xyz)
    out_n[len(sfm_xyz):] = dsm_n

    n_all = out_xyz.shape[0]
    confidence = np.zeros((n_all,), dtype=np.float32)
    anchor_id = np.full((n_all,), -1, dtype=np.int32)
    is_dsm = np.zeros((n_all,), dtype=bool)
    rotations = np.zeros((n_all, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    confidence[len(sfm_xyz):] = dsm_c.astype(np.float32)
    anchor_id[len(sfm_xyz):] = dsm_aid.astype(np.int32)
    is_dsm[len(sfm_xyz):] = True
    for i, nvec in enumerate(dsm_n):
        rotations[len(sfm_xyz) + i] = quat_from_normal(nvec)

    meta = {
        "confidence": confidence,
        "anchor_id": anchor_id,
        "is_dsm": is_dsm,
        "rotations": rotations,
        "tangent_scale": tang,
        "normal_scale": nscale,
        "clone_conf_scale": float(getattr(opt, "dsm_clone_conf_scale", 0.5)),
        "stats": {
            "n_sfm_in": int(xyz.shape[0]),
            "n_near_surface": n_replace,
            "n_floaters_removed": int(floaters.sum()),
            "n_sfm_kept": int(keep_sfm.sum()),
            "n_dsm": n_dsm,
            "n_out": n_all,
        },
    }
    print(f"[DSM init] {meta['stats']}")
    pcd_out = BasicPointCloud(points=out_xyz.astype(np.float32), colors=out_rgb.astype(np.float32), normals=out_n.astype(np.float32))
    return pcd_out, meta

def add_dsm_losses(loss, gaussians, prior, opt, iteration, is_idu=False):
    """Source C surface term. Stage 2 never implicitly enables this loss."""
    if is_idu or prior is None or not getattr(opt, 'dsm_surface', True):
        return loss, {}
    coefficient = float(getattr(opt, 'lambda_dsm_surface', 0.05)) * dsm_loss_scale(iteration, opt)
    if coefficient <= 0 or not gaussians.dsm_confidence.numel():
        return loss, {}
    xyz = gaussians.get_xyz
    idx = torch.where(gaussians.dsm_confidence > 0)[0]
    limit = int(getattr(opt, 'dsm_surface_sample', 65536))
    if idx.numel() > limit:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)[:limit]]
    if not idx.numel():
        return loss, {'dsm_surface': 0.0, 'dsm_active_points': 0}
    h, n, confidence = prior.sample(xyz[idx, :2])
    alpha = gaussians.get_opacity.detach().squeeze(-1).to(xyz.dtype).clamp_min(1e-6)
    weight = confidence.detach() * alpha[idx] * gaussians.dsm_confidence[idx]
    p = torch.stack([xyz[idx, 0], xyz[idx, 1], h], dim=-1)
    residual = ((xyz[idx] - p) * n.detach()).sum(dim=-1)
    sigma = float(getattr(opt, 'dsm_surface_sigma', 0.5))
    surface = (weight * torch.log1p((residual / sigma)**2)).sum() / weight.sum().clamp_min(1e-8)
    return loss + coefficient * surface, {
        'dsm_surface': float(surface.detach()), 'dsm_coefficient': coefficient,
        'dsm_active_points': int((weight > 0).sum()),
    }
