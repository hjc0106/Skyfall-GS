from __future__ import annotations

from pathlib import Path

import numpy as np

from prf.types import PinholeView



def expected_depth_from_accumulated(depth: np.ndarray, alpha: np.ndarray, *, min_alpha: float = 1e-6) -> np.ndarray:
    """Convert gsplat RGB+D accumulated z (sum w_i z_i) to expected z (sum w_i z_i / sum w_i)."""
    denom = np.maximum(np.asarray(alpha, dtype=np.float64), min_alpha)
    expected = np.asarray(depth, dtype=np.float64) / denom
    expected[np.asarray(alpha) < min_alpha] = np.nan
    return expected.astype(np.float32, copy=False)


def render_training_depth_maps(
    ply_path: Path | str,
    views: list[PinholeView],
    *,
    device: str = "cuda",
    cache_dir: Path | str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rasterize expected camera-space z/alpha directly with gsplat.

    Complete caches can be consumed without torch, gsplat, or a GPU.
    """

    cache = Path(cache_dir) if cache_dir is not None else None
    maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing: list[PinholeView] = []
    for view in views:
        cached = None if cache is None else cache / f"{view.view_id}.npz"
        if cached is not None and cached.is_file():
            payload = np.load(cached)
            depth = payload["depth"]
            alpha = payload["alpha"]
            kind = str(payload["depth_kind"][0]) if "depth_kind" in payload.files else "accumulated"
            if kind != "expected":
                depth = expected_depth_from_accumulated(depth, alpha)
                np.savez_compressed(
                    cached,
                    depth=depth,
                    alpha=alpha,
                    depth_kind=np.array(["expected"]),
                )
            maps[view.view_id] = (depth, alpha)
        else:
            missing.append(view)
    if not missing:
        return maps

    import torch
    try:
        from gsplat import rasterization
    except ImportError as exc:
        raise ImportError("GS-depth visibility requires gsplat; install requirements-depth.txt") from exc
    from prf.io_gs import load_gaussians_ply

    torch_device = torch.device(device)
    scene = load_gaussians_ply(ply_path)
    def tensor(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=torch_device).contiguous()

    means = tensor(scene.mu)
    quats = tensor(scene.rotations)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = tensor(scene.scales)
    opacities = tensor(scene.opacity)
    # RGB does not affect accumulated depth or alpha; SH coefficients are unnecessary.
    colors = torch.zeros_like(means)
    with torch.inference_mode():
        for view in missing:
            rendered, alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=tensor(view.w2c).unsqueeze(0),
                Ks=tensor([[view.fx, 0.0, view.cx],
                           [0.0, view.fy, view.cy], [0.0, 0.0, 1.0]]).unsqueeze(0),
                width=int(view.width),
                height=int(view.height),
                backgrounds=torch.ones(1, 3, device=torch_device),
                render_mode="RGB+D",
                packed=False,
                sh_degree=None,
            )
            alpha = alphas[0, ..., 0].detach().float().cpu().numpy()
            accumulated = rendered[0, ..., 3].detach().float().cpu().numpy()
            depth = expected_depth_from_accumulated(accumulated, alpha)
            maps[view.view_id] = (depth, alpha)
            print(
                f"  {view.view_id}: depth {depth.shape[1]}x{depth.shape[0]} "
                f"alpha_mean={float(alpha.mean()):.3f}",
                flush=True,
            )
            if cache is not None:
                cache.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache / f"{view.view_id}.npz",
                    depth=depth,
                    alpha=alpha,
                    depth_kind=np.array(["expected"]),
                )
    return maps
