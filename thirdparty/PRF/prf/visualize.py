"""Paint Physical Resolution Field values onto images and a top-down map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prf.cameras import load_training_views
from prf.types import PinholeView

BOTTLENECK_BGR = {
    "KERNEL": (40, 140, 255),
    "SPACING": (220, 160, 40),
    "OBS": (60, 200, 80),
    "INVALID": (120, 120, 120),
}


def load_field(field_dir: Path) -> dict[str, np.ndarray]:
    data = np.load(field_dir / "physical_resolution_field.npz", allow_pickle=True)
    return {key: data[key] for key in data.files}


def turbo_rgb(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scale = max(vmax - vmin, 1e-12)
    idx = np.clip((values - vmin) / scale, 0.0, 1.0)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
    lut = cv2.cvtColor(lut, cv2.COLOR_BGR2RGB).reshape(256, 3)
    return lut[(idx * 255.0).astype(np.uint8)]


def colorbar_rgb(vmin: float, vmax: float, height: int = 256, width: int = 72, unit: str = "m") -> np.ndarray:
    ramp = np.linspace(vmax, vmin, height, dtype=np.float64)
    colors = turbo_rgb(ramp, vmin, vmax)
    bar = np.repeat(colors[:, None, :], max(width - 44, 8), axis=1)
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    canvas[:, : bar.shape[1]] = bar
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for frac, label in ((0.0, f"{vmax:.2g}{unit}"), (0.5, f"{0.5 * (vmin + vmax):.2g}"), (1.0, f"{vmin:.2g}{unit}")):
        y = int(frac * (height - 1))
        draw.text((bar.shape[1] + 4, max(0, y - 6)), label, fill=(230, 230, 230), font=font)
    return np.asarray(image)


def _splat_xy(
    height: int,
    width: int,
    uv: np.ndarray,
    values: np.ndarray,
    radius: int,
) -> np.ndarray:
    acc = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    radius = max(int(radius), 1)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (xx * xx + yy * yy) <= radius * radius
    for (u, v), value in zip(uv, values, strict=True):
        if not np.isfinite(value):
            continue
        x0 = int(round(u))
        y0 = int(round(v))
        y1 = y0 - radius
        y2 = y0 + radius + 1
        x1 = x0 - radius
        x2 = x0 + radius + 1
        gy1, gy2 = max(0, y1), min(height, y2)
        gx1, gx2 = max(0, x1), min(width, x2)
        if gy1 >= gy2 or gx1 >= gx2:
            continue
        patch = disk[gy1 - y1 : gy2 - y1, gx1 - x1 : gx2 - x1]
        acc[gy1:gy2, gx1:gx2][patch] += value
        weight[gy1:gy2, gx1:gx2][patch] += 1.0
    out = np.full((height, width), np.nan, dtype=np.float64)
    seen = weight > 0
    out[seen] = acc[seen] / weight[seen]
    return out


def topdown_map(
    centers: np.ndarray,
    values: np.ndarray,
    *,
    size: int = 1024,
    radius_m: float = 12.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    finite = np.isfinite(values)
    xy = centers[finite, :2]
    vals = values[finite]
    lo = xy.min(axis=0) - radius_m
    hi = xy.max(axis=0) + radius_m
    span = np.maximum(hi - lo, 1.0)
    scale = (size - 1) / span.max()
    uv = (xy - lo) * scale
    uv[:, 1] = (size - 1) - uv[:, 1]
    raster = _splat_xy(size, size, uv, vals, radius=max(int(round(radius_m * scale)), 2))
    xmin, ymin = lo.tolist()
    xmax, ymax = hi.tolist()
    return raster, (xmin, xmax, ymin, ymax)


def overlay_points(
    rgb: np.ndarray,
    uv: np.ndarray,
    colors: np.ndarray,
    *,
    radius: np.ndarray | int,
    alpha: float = 0.58,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    image = rgb.copy()
    height, width = image.shape[:2]
    overlay = image.copy()
    if valid is None:
        valid = np.ones(len(uv), dtype=bool)
    if np.isscalar(radius):
        radii = np.full(len(uv), int(radius), dtype=np.int32)
    else:
        radii = np.asarray(radius, dtype=np.int32)
    for (u, v), color, keep, rad in zip(uv, colors, valid, radii, strict=True):
        if not keep or not np.isfinite(u) or not np.isfinite(v):
            continue
        x, y = int(round(u)), int(round(v))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        cv2.circle(
            overlay,
            (x, y),
            max(int(rad), 2),
            (int(color[2]), int(color[1]), int(color[0])),
            -1,
            lineType=cv2.LINE_AA,
        )
    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def match_view(views: list[PinholeView], image_path: Path) -> PinholeView:
    stem = image_path.stem
    name = image_path.name
    hits = [view for view in views if view.file_path.endswith(name) or Path(view.file_path).name == name]
    if len(hits) == 1:
        return hits[0]
    hits = [view for view in views if view.view_id == stem]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return hits[0]
    known = ", ".join(view.file_path or view.view_id for view in views[:8])
    raise KeyError(f"No camera for {image_path}. Known: {known}")


def attach_colorbar(rgb: np.ndarray, vmin: float, vmax: float, unit: str = "m") -> np.ndarray:
    bar = colorbar_rgb(vmin, vmax, height=rgb.shape[0], unit=unit)
    return np.concatenate([rgb, bar], axis=1)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def write_colored_ply(path: Path, centers: np.ndarray, colors: np.ndarray) -> None:
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(centers)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    lines = [header]
    for xyz, rgb in zip(centers, colors, strict=True):
        lines.append(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")
    path.write_text("".join(lines))


def _panel(images: list[np.ndarray], titles: list[str]) -> np.ndarray:
    height = min(im.shape[0] for im in images)
    resized = []
    font = ImageFont.load_default()
    for image, title in zip(images, titles, strict=True):
        if image.shape[0] != height:
            scale = height / image.shape[0]
            image = cv2.resize(image, (int(image.shape[1] * scale), height), interpolation=cv2.INTER_AREA)
        canvas = np.full((height + 22, image.shape[1], 3), 18, dtype=np.uint8)
        canvas[22:] = image
        pil = Image.fromarray(canvas)
        ImageDraw.Draw(pil).text((8, 4), title, fill=(230, 230, 230), font=font)
        resized.append(np.asarray(pil))
    return np.concatenate(resized, axis=1)


def _finite_scale(values: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    return float(np.quantile(finite, 0.05)), float(np.quantile(finite, 0.95))


def render_overlays(
    field: dict[str, np.ndarray],
    view: PinholeView,
    rgb: np.ndarray,
    *,
    r_phys_range: tuple[float, float],
    r_obs_range: tuple[float, float],
    r_kernel_range: tuple[float, float],
) -> dict[str, np.ndarray]:
    view.width, view.height = rgb.shape[1], rgb.shape[0]
    uv, z = view.project(field["center"])
    in_front = np.isfinite(z) & (z > 1.0)
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0])
    keep = in_front & in_image
    radius_px = np.clip(12.0 * view.fx / np.maximum(z, 1.0), 3.0, 28.0)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = {}
    specs = [
        ("r_phys", field["R_phys"], r_phys_range),
        ("r_obs", field["R_obs"], r_obs_range),
        ("r_kernel", field["R_kernel"], r_kernel_range),
    ]
    for name, values, lohi in specs:
        colors = turbo_rgb(np.where(np.isfinite(values), values, lohi[1]), *lohi)
        overlay = overlay_points(bgr, uv, colors, radius=radius_px, valid=keep & np.isfinite(values))
        out[name] = attach_colorbar(overlay, *lohi)
    labels = field["bottleneck"].astype(str)
    colors = np.array([BOTTLENECK_BGR.get(name, BOTTLENECK_BGR["INVALID"])[::-1] for name in labels], dtype=np.uint8)
    out["bottleneck"] = overlay_points(bgr, uv, colors, radius=radius_px, valid=keep)
    return out


def write_html(output: Path, cards: list[tuple[str, str, str]]) -> None:
    items = "\n".join(
        f'<figure><img src="{src}" alt="{title}"><figcaption><strong>{title}</strong><br>{caption}</figcaption></figure>'
        for title, src, caption in cards
    )
    output.write_text(
        """<!doctype html>
<meta charset="utf-8">
<title>PRF 可视化</title>
<style>
body { font-family: sans-serif; background: #111; color: #eee; margin: 24px; }
figure { margin: 0 0 32px; }
img { max-width: 100%; height: auto; background: #000; }
figcaption { margin-top: 8px; color: #ccc; line-height: 1.4; }
h1 { font-size: 20px; }
p { color: #aaa; }
</style>
<h1>Physical Resolution Field 可视化</h1>
<p>颜色越蓝越细，越红越粗。R 单位是 m/equiv.pixel。</p>
"""
        + items
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay PRF values on RGB images and a top-down map.")
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, action="append", default=[])
    parser.add_argument("--opengl-c2w", action="store_true")
    parser.add_argument("--image", type=Path, action="append", default=[], help="RGB image to overlay; camera is matched from --transforms")
    parser.add_argument("--preview-root", type=Path, help="Directory containing transforms_opengl.json and renders/")
    parser.add_argument("--preview-frame", action="append", default=[], help="Relative render path under --preview-root")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    field = load_field(args.field)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    r_phys_range = _finite_scale(field["R_phys"], (3.0, 16.0))
    r_obs_range = _finite_scale(field["R_obs"], (0.3, 1.2))
    r_kernel_range = _finite_scale(field["R_kernel"], (3.0, 16.0))

    cards: list[tuple[str, str, str]] = []
    for name, values, lohi, caption in (
        ("topdown_r_phys.png", field["R_phys"], r_phys_range, "俯视 R_phys。蓝细红粗。"),
        ("topdown_r_obs.png", field["R_obs"], r_obs_range, "俯视 R_obs（原始训练相机采样）。"),
        ("topdown_r_kernel.png", field["R_kernel"], r_kernel_range, "俯视 R_kernel（Gaussian 核尺度）。"),
    ):
        raster, _ = topdown_map(field["center"], values)
        colors = turbo_rgb(np.where(np.isfinite(raster), raster, lohi[1]), *lohi)
        colors[~np.isfinite(raster)] = 18
        rgb = attach_colorbar(colors, *lohi)
        save_rgb(output / name, rgb)
        cards.append((name.replace(".png", ""), name, caption))

    colors = turbo_rgb(np.where(np.isfinite(field["R_phys"]), field["R_phys"], r_phys_range[1]), *r_phys_range)
    write_colored_ply(output / "physical_resolution_field_colored.ply", field["center"], colors)

    views: list[PinholeView] = []
    for path in args.transforms:
        views.extend(load_training_views(path, opengl_c2w=args.opengl_c2w))
    image_jobs: list[tuple[Path, list[PinholeView], str]] = [(path, views, path.stem) for path in args.image]
    if args.preview_root:
        preview_views = load_training_views(args.preview_root / "transforms_opengl.json", opengl_c2w=True)
        for rel in args.preview_frame:
            image_jobs.append((args.preview_root / rel, preview_views, Path(rel).parent.name + "_" + Path(rel).stem))

    for image_path, camera_views, stem in image_jobs:
        view = match_view(camera_views, image_path)
        rgb = load_rgb(image_path)
        overlays = render_overlays(
            field,
            view,
            rgb,
            r_phys_range=r_phys_range,
            r_obs_range=r_obs_range,
            r_kernel_range=r_kernel_range,
        )
        save_rgb(output / f"{stem}_overlay_r_phys.png", overlays["r_phys"])
        save_rgb(output / f"{stem}_overlay_r_obs.png", overlays["r_obs"])
        save_rgb(output / f"{stem}_overlay_r_kernel.png", overlays["r_kernel"])
        save_rgb(output / f"{stem}_overlay_bottleneck.png", overlays["bottleneck"])
        panel = _panel(
            [rgb, overlays["r_phys"][:, : rgb.shape[1]], overlays["r_obs"][:, : rgb.shape[1]], overlays["r_kernel"][:, : rgb.shape[1]]],
            ["RGB", "R_phys", "R_obs", "R_kernel"],
        )
        save_rgb(output / f"{stem}_panel.png", panel)
        cards.append((f"{stem} R_phys", f"{stem}_overlay_r_phys.png", f"叠在 {image_path.name} 上的 R_phys"))
        cards.append((f"{stem} 四联图", f"{stem}_panel.png", "从左到右：原图、R_phys、R_obs、R_kernel"))

    write_html(output / "index.html", cards)
    (output / "scale.json").write_text(
        json.dumps(
            {"R_phys": r_phys_range, "R_obs": r_obs_range, "R_kernel": r_kernel_range},
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote visualizations to {output}")


if __name__ == "__main__":
    main()
