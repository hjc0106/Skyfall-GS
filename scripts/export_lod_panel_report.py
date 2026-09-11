#!/usr/bin/env python3
"""Build the JAX_068 2x->4x 3-ROI results page. Does not train or add experiments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

CJK_FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

from lod.lineage import write_json
from lod.panel import PANEL_DIR, PANEL_ROIS, memory_accounting, summarize_panel


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "building": "建筑边界",
    "trees": "树木",
    "parking": "停车场",
}
PLOT_LABELS = {
    "building": "building",
    "trees": "trees",
    "parking": "parking",
}
COLORS = {"building": "#c23b22", "trees": "#2a7d4f", "parking": "#2c5aa0"}


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _open(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _fit(image: Image.Image, size: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    canvas.paste(copy, ((size - copy.width) // 2, (size - copy.height) // 2))
    return canvas


def _center_roi_crop(image: Image.Image, zoom: float, roi_size: float = 0.1) -> Image.Image:
    frac = roi_size * zoom
    frac = min(1.0, max(0.05, frac))
    width, height = image.size
    crop_w = max(1, int(round(width * frac)))
    crop_h = max(1, int(round(height * frac)))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return image.crop((left, top, left + crop_w, top + crop_h))


def _cell(image: Image.Image, caption: str, size: int) -> Image.Image:
    thumb = _fit(image, size)
    out = Image.new("RGB", (size, size + 28), (255, 255, 255))
    out.paste(thumb, (0, 28))
    draw = ImageDraw.Draw(out)
    font = ImageFont.truetype(str(CJK_FONT), 14) if CJK_FONT.is_file() else ImageFont.load_default()
    draw.text((8, 8), caption, fill=(40, 40, 40), font=font)
    return out


def _montage(panel: Path, roi_id: str, out_path: Path, cell: int = 280) -> None:
    l1 = panel / roi_id / "l1"
    l2 = panel / roi_id / "l2"
    pairs = [
        ("2× 冻结输入", _open(l1 / "render_input.png"), 2.0, False),
        ("2× 超分监督", _open(l1 / "refined.png"), 2.0, False),
        ("2× 训练后", _open(l1 / "steps" / "0500" / "target.png"), 2.0, False),
        ("2× 局部·输入", _center_roi_crop(_open(l1 / "render_input.png"), 2.0), 2.0, True),
        ("2× 局部·监督", _center_roi_crop(_open(l1 / "refined.png"), 2.0), 2.0, True),
        ("2× 局部·训练后", _center_roi_crop(_open(l1 / "steps" / "0500" / "target.png"), 2.0), 2.0, True),
        ("4× 冻结输入", _open(l2 / "render_input.png"), 4.0, False),
        ("4× 超分监督", _open(l2 / "refined.png"), 4.0, False),
        ("4× 训练后", _open(l2 / "steps" / "0500" / "target.png"), 4.0, False),
        ("4× 局部·输入", _center_roi_crop(_open(l2 / "render_input.png"), 4.0), 4.0, True),
        ("4× 局部·监督", _center_roi_crop(_open(l2 / "refined.png"), 4.0), 4.0, True),
        ("4× 局部·训练后", _center_roi_crop(_open(l2 / "steps" / "0500" / "target.png"), 4.0), 4.0, True),
    ]
    cells = [_cell(image, caption, cell) for caption, image, _, _ in pairs]
    cols, rows = 3, 4
    pad = 8
    tile_w, tile_h = cells[0].size
    canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * pad, rows * tile_h + (rows + 1) * pad), (255, 255, 255))
    for index, tile in enumerate(cells):
        x = pad + (index % cols) * (tile_w + pad)
        y = pad + (index // cols) * (tile_h + pad)
        canvas.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _curve_series(curve: list[dict], *, rgb=True, parent=False, train_gt=False):
    xs, ys = [], []
    for item in curve:
        xs.append(int(item["step"]))
        if parent:
            ys.append(((item.get("parent_2x") or {}).get("l1_vs_frozen_l1")))
        elif train_gt:
            ys.append(((item.get("train_views_mean") or {}).get("l1_to_gt_mean")))
        else:
            key = "l1_to_refined" if rgb else "hf_l1_to_refined"
            ys.append((item.get("target") or {}).get(key))
    return xs, ys


def _plot_curves(panel: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), dpi=140)
    specs = [
        (axes[0, 0], "l1", True, "L1 RGB MAE vs supervision"),
        (axes[0, 1], "l1", False, "L1 Laplacian MAE vs supervision"),
        (axes[1, 0], "l2", True, "L2 RGB MAE vs supervision"),
        (axes[1, 1], "l2", False, "L2 Laplacian MAE vs supervision"),
    ]
    for ax, level, rgb, title in specs:
        for item in PANEL_ROIS:
            data = _load(panel / item["id"] / level / "absorption_curve.json")
            xs, ys = _curve_series(list(data.get("curve") or []), rgb=rgb)
            ax.plot(xs, ys, marker="o", color=COLORS[item["id"]], label=PLOT_LABELS[item["id"]], linewidth=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_absorption.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=140)
    for item in PANEL_ROIS:
        data = _load(panel / item["id"] / "l2" / "absorption_curve.json")
        curve = list(data.get("curve") or [])
        xs, parent = _curve_series(curve, parent=True)
        _, gt = _curve_series(curve, train_gt=True)
        axes[0].plot(xs, parent, marker="o", color=COLORS[item["id"]], label=PLOT_LABELS[item["id"]], linewidth=1.6)
        axes[1].plot(xs, gt, marker="o", color=COLORS[item["id"]], label=PLOT_LABELS[item["id"]], linewidth=1.6)
    axes[0].set_title("2x RGB MAE vs frozen L1", fontsize=10)
    axes[1].set_title("1x train-view RGB MAE vs GT", fontsize=10)
    axes[1].set_ylim(0.0185, 0.0195)
    for ax in axes:
        ax.set_xlabel("L2 step")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "curves_old_scale.png")
    plt.close(fig)


def _rel(before, after) -> float | None:
    if before in (None, 0):
        return None
    return (after - before) / before


def _fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _pct(rel) -> str:
    if rel is None:
        return "—"
    return f"{rel * 100:+.1f}%"


def build_html(*, panel: Path, figures: Path, mem: dict, summary: dict, selection: dict) -> str:
    attempts = _load(panel / "ATTEMPTS.json").get("failed_attempts") or []
    smi = mem["nvidia_smi_memory_used"]
    worker = mem["dloral_worker_allocated"]
    rows = []
    for item in PANEL_ROIS:
        roi = next(row for row in summary["rois"] if row["id"] == item["id"])
        l1 = roi["supervision_absorption"]["l1"]
        l2 = roi["supervision_absorption"]["l2"]
        res = roi["resources"]
        old = roi["old_scale"]
        vis = roi.get("visual_quality") or {}
        zoom = (roi.get("correspondence_and_transition") or {}).get("continuous_zoom") or {}
        if not vis.get("passed"):
            vis = {**vis, "passed": zoom.get("passed"), "unresolved": zoom.get("unresolved"),
                   "status": vis.get("status") or zoom.get("status")}
        corr = (roi.get("correspondence_and_transition") or {}).get("coverage")
        acc = _load(panel / item["id"] / "ACCEPTANCE.json")
        rec = acc.get("recover") or {}
        rows.append({
            "id": item["id"],
            "label": LABELS[item["id"]],
            "roi": item,
            "l1": l1,
            "l2": l2,
            "res": res,
            "old": old,
            "vis": vis,
            "corr": corr,
            "stage_link": (acc.get("checks") or {}).get("stage_link") or {},
            "freeze": rec.get("freeze_ok"),
            "l1_rgb_rel": _rel(l1["rgb_mae"]["0"], l1["rgb_mae"]["500"]),
            "l2_rgb_rel": _rel(l2["rgb_mae"]["0"], l2["rgb_mae"]["500"]),
            "l2_hf_rel": _rel(l2["laplacian_mae"]["0"], l2["laplacian_mae"]["500"]),
        })

    attempt_html = "".join(
        f"<li><code>{html.escape(item['roi'])}</code> @ ({item['center']['center_x']}, {item['center']['center_y']})："
        f"{html.escape(item['reason'])}；{html.escape(item['recovery'])}</li>"
        for item in attempts
    ) or "<li>无</li>"

    def roi_section(row):
        rid = row["id"]
        vis = row["vis"]
        passed = "".join(f"<li>{html.escape(text)}</li>" for text in (vis.get("passed") or []) or [])
        unresolved = "".join(f"<li>{html.escape(text)}</li>" for text in (vis.get("unresolved") or []) or [])
        return f"""
<section>
<h3>{row['label']} <code>({row['roi']['center_x']:.2f}, {row['roi']['center_y']:.2f})</code></h3>
<p class="note">{html.escape(row['roi']['rationale'])}</p>
<figure>
<img src="figures/montage_{rid}.png" alt="{row['label']} 输入、监督与训练后对照">
<figcaption>上两行 2×，下两行 4×。局部图是注册 ROI 在该尺度可见窗中的中心裁剪，不是事后按误差选框。</figcaption>
</figure>
<figure>
<img src="figures/zoom_{rid}_start.png" alt="{row['label']} 缩放起点">
<img src="figures/zoom_{rid}_end.png" alt="{row['label']} 缩放终点">
<figcaption>连续缩放视频的起、末抽帧。仅作 visual_checked，不作为 video_fully_passed。</figcaption>
</figure>
<div class="split">
<div>
<p><b>执行</b> 来源完整；阶段衔接 PNG L1={_fmt(row['stage_link'].get('rgb_l1_vs_saved_4x_input'), 4)}；freeze_ok={row['freeze']}。</p>
<p><b>吸收</b> L1 RGB {_fmt(row['l1']['rgb_mae']['0'], 4)}→{_fmt(row['l1']['rgb_mae']['500'], 4)}（{_pct(row['l1_rgb_rel'])}）；
L2 RGB {_fmt(row['l2']['rgb_mae']['0'], 4)}→{_fmt(row['l2']['rgb_mae']['500'], 4)}（{_pct(row['l2_rgb_rel'])}）；
L2 Laplacian {_fmt(row['l2']['laplacian_mae']['0'], 4)}→{_fmt(row['l2']['laplacian_mae']['500'], 4)}（{_pct(row['l2_hf_rel'])}）。</p>
<p><b>旧尺度</b> 1× vs GT {_fmt(row['old']['train_gt_1x'], 8)}；2× vs 冻结 L1 {_fmt(row['old']['parent_2x_vs_frozen_l1'], 5)}；共可见覆盖 {_fmt(row['corr'], 3)}。</p>
</div>
<div>
<p><b>视觉</b> {html.escape(str(vis.get('status') or 'pending_visual'))}，不是 <code>video_fully_passed</code>。</p>
<p>通过项</p><ul>{passed or '<li>未记</li>'}</ul>
<p>未决项</p><ul>{unresolved or '<li>未记</li>'}</ul>
</div>
</div>
</section>
"""

    body_rois = "\n".join(roi_section(row) for row in rows)
    table_rows = "\n".join(
        f"<tr><td>{row['label']}</td>"
        f"<td>{_fmt(row['l1']['rgb_mae']['0'], 4)} → {_fmt(row['l1']['rgb_mae']['500'], 4)} ({_pct(row['l1_rgb_rel'])})</td>"
        f"<td>{_fmt(row['l2']['rgb_mae']['0'], 4)} → {_fmt(row['l2']['rgb_mae']['500'], 4)} ({_pct(row['l2_rgb_rel'])})</td>"
        f"<td>{_fmt(row['l2']['laplacian_mae']['0'], 4)} → {_fmt(row['l2']['laplacian_mae']['500'], 4)} ({_pct(row['l2_hf_rel'])})</td>"
        f"<td>{_fmt(row['old']['train_gt_1x'], 8)}</td>"
        f"<td>{_fmt(row['old']['parent_2x_vs_frozen_l1'], 5)}</td>"
        f"<td>{_fmt(row['res']['generate_s'], 1)} / {_fmt(row['res']['train_s'], 1)}</td>"
        f"<td>{row['res']['n_l1']} / {row['res']['n_l2']}</td></tr>"
        for row in rows
    )
    smi_peak = smi.get("peak_mib")
    smi_display = smi.get("display") or (f"{smi_peak:,} MiB ≈ {smi.get('peak_gib'):.2f} GiB" if smi_peak else "—")
    worker_display = worker.get("display") or "—"
    worker_bytes = worker.get("peak_bytes")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JAX_068 场景内小样本稳定性与监督吸收</title>
<style>
:root {{ font-family: "Source Han Sans SC", "Noto Sans CJK SC", "PingFang SC", sans-serif; color: #1f1f1f; background: #f4f1ea; }}
body {{ margin: 0; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }}
h1, h2, h3 {{ font-weight: 650; letter-spacing: 0.02em; }}
h1 {{ font-size: 28px; margin-bottom: 8px; }}
h2 {{ margin-top: 36px; border-bottom: 1px solid #d7d0c4; padding-bottom: 6px; }}
.eyebrow {{ color: #6b6458; font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; }}
.claim {{ background: #fff; border: 1px solid #e2dbd0; padding: 16px 18px; border-radius: 10px; }}
.note, figcaption {{ color: #5c564c; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }}
th, td {{ border: 1px solid #e2dbd0; padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f8f4ec; }}
figure {{ margin: 12px 0 24px; }}
img {{ max-width: 100%; height: auto; background: #fff; border: 1px solid #e2dbd0; }}
.split {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; }}
ul {{ margin: 6px 0 12px 18px; }}
code {{ font-size: 12px; }}
@media (max-width: 800px) {{ .split {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<p class="eyebrow">LoD scale chain · JAX_068 · view 0 · closed</p>
<h1>JAX_068 场景内小样本稳定性与监督吸收验证</h1>
<div class="claim">
<p><b>正式收口。</b>固定设置 2×→4× 已在 JAX_068 同一基础视图的建筑 / 树木 / 停车场三类 ROI 上完成新来源链。结果页同时覆盖效果、旧尺度影响、资源开销和验收边界。结论限于<strong>场景内小样本稳定性与监督吸收</strong>。</p>
<p><b>证据支持：</b>三类内容均能吸收监督；树木高频改善相对有限；旧尺度变化较小。</p>
<p><b>证据不支持：</b>生成真实性、跨场景泛化、geometry 优势。视频闪烁继续保留未验，不是 <code>video_fully_passed</code>。</p>
<p>未做 SpyNet 全套对照、未上 8×、未处理低优先级局部重投影。无需在 JAX_068 继续增加相似 ROI，也无需改算法。若继续，优先另一个场景的一条完整 2×→4× 来源链：提前选定 ROI，保持现有设置，先验证 Stage1／相机／深度适配，再完成生成、训练和同口径验收。</p>
</div>

<h2>1. 设置与选点</h2>
<p>geometry 对齐，seed=0，每层 500 步，点数预算 50k，mix=0.2，增密 1–250 / interval=10，往返门控关闭。基础视图为 train cameras[0] = <code>JAX_068_011_RGB</code>。选择清单先于训练落盘。</p>
<figure>
<img src="figures/overlay_view0.png" alt="三个预登记 ROI 与排除的历史框">
<figcaption>红：建筑边界 (0.38, 0.50)。绿：树木 (0.25, 0.48)。蓝：停车场 (0.72, 0.68)。灰框为未使用的历史中心。</figcaption>
</figure>
<p class="note">树木最初锁在 (0.20, 0.48)，2× 水平越界。改正到 x=0.25 发生在树木与停车场开训之前，仍是同一片树冠，不是按指标重采样。</p>

<h2>2. 执行记录</h2>
<ul>
<li>三个最终有效配置全部完成新来源链，<code>source_complete: true</code>。</li>
<li>另有一次初始 ROI 越界失败，属于选点修正，不是断点恢复。</li>
<li>历史 <code>jax068_ybuilding</code> / <code>jax068_0p28_0p28</code> 仍为 <code>passed_small_scope</code> / <code>regression_sample</code>，<code>source_complete: false</code>。</li>
<li>短程恢复只写成数值近似一致（xyz 最大差 3.8×10<sup>−6</sup>）；12→16 未跨过下一次增密事件。</li>
</ul>
<ul>{attempt_html}</ul>

<h2>3. 吸收与旧尺度</h2>
<table>
<thead><tr><th>ROI</th><th>L1 RGB</th><th>L2 RGB</th><th>L2 Laplacian</th><th>1× vs GT</th><th>2× vs 冻结 L1</th><th>生成/训练 s</th><th>n L1 / L2</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
<ul>
<li><b>吸收表现一致。</b>三组 L1 RGB 均下降约 37%–38%；L2 RGB 下降约 43%–54%。没有某类区域完全无法拟合。</li>
<li><b>树木高频吸收相对有限。</b>L2 Laplacian 相对下降约为建筑 17%、树木 9%、停车场 14%。这是内容差异线索，不能归因于单一模块。</li>
<li><b>旧尺度影响稳定。</b>2× 相对冻结 L1 为 0.0023–0.0025。1× 对 GT 的均值在 0 步与 500 步按 1e-8 门控记为不变；若要声明“完全不变”，应引用该直接差分或 freeze 检查，而不是四位舍入后的 0.0190。</li>
</ul>
<figure>
<img src="figures/curves_absorption.png" alt="三 ROI 的 RGB 与 Laplacian 吸收曲线">
<figcaption>吸收曲线。L2 在 step 50 可暂时变差，与点预算填满有关；500 步后相对 step 0 均下降。</figcaption>
</figure>
<figure>
<img src="figures/curves_old_scale.png" alt="旧尺度 1× 与 2× 变化">
<figcaption>左：2× 相对冻结 L1。右：训练视角 1× 对 GT，纵轴固定在 0.0185–0.0195，避免把 1e-10 量级的舍入差画成曲线分离。两条图纵轴尺度不同，不要比成同一量。若要声明 1×「完全不变」，引用 0/500 步 &lt; 1e-8 差分或 freeze，而不是四位舍入。</figcaption>
</figure>

<h2>4. 图像对照</h2>
{body_rois}

<h2>5. 资源与显存口径</h2>
<p>单位写全，避免再次把整卡占用与 worker 张量分配混成一个数。</p>
<table>
<thead><tr><th>口径</th><th>数值</th><th>来源</th><th>定义</th></tr></thead>
<tbody>
<tr><td>整卡峰值</td><td><b>{html.escape(smi_display)}</b></td><td><code>nvidia-smi memory.used</code></td><td>{html.escape(smi['note'])}</td></tr>
<tr><td>worker 张量分配峰值</td><td><b>{html.escape(worker_display)}</b></td><td><code>max_memory_allocated</code>（{worker_bytes} B）</td><td>{html.escape(worker['note'])}</td></tr>
<tr><td>L1 / L2 训练整卡</td><td>约 19–25 GiB</td><td><code>nvidia-smi memory.used</code></td><td>仅 3DGS 训练阶段，当时无 DLoRAL worker</td></tr>
</tbody>
</table>
<p class="note">两者测量范围不同，不能相减后直接解释成父进程占用，也不能把差值读成性能退化。</p>

<h2>6. 不在本页范围</h2>
<ul>
<li>生成真实性、跨场景泛化、geometry 优势。</li>
<li>视频闪烁；三个 ROI 都来自 JAX_068 的同一基础视图。</li>
<li>局部重投影优化、8×、改增密、加尺度约束，或在 JAX_068 再加相似 ROI。</li>
</ul>
<p class="note">数据：<code>skyfall-gs_exp/lod_panel_3roi</code>。索引：<code>skyfall-gs_exp/lod_index.json</code>。本页由 <code>scripts/export_lod_panel_report.py</code> 生成。</p>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel_dir", type=str, default=str(PANEL_DIR))
    parser.add_argument("--out_dir", type=str, default="docs/lod_panel_3roi")
    args = parser.parse_args()
    panel = Path(args.panel_dir)
    if not panel.is_absolute():
        panel = (ROOT / panel).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    overlay = panel / "selection" / "overlay_view0.png"
    if overlay.is_file():
        _fit(_open(overlay), 1024).save(figures / "overlay_view0.png")
    for item in PANEL_ROIS:
        _montage(panel, item["id"], figures / f"montage_{item['id']}.png")
        frames = sorted((panel / item["id"] / "zoom_frames").glob("frame_*.png"))
        if frames:
            _fit(_open(frames[0]), 640).save(figures / f"zoom_{item['id']}_start.png")
            _fit(_open(frames[-1]), 640).save(figures / f"zoom_{item['id']}_end.png")
    _plot_curves(panel, figures)

    mem = memory_accounting(panel)
    selection = _load(panel / "SELECTION.json")
    summary = summarize_panel(panel, selection=selection)
    summary["memory_accounting"] = {
        "nvidia_smi_peak_mib": mem["nvidia_smi_memory_used"]["peak_mib"],
        "nvidia_smi_display": mem["nvidia_smi_memory_used"]["display"],
        "worker_allocated_bytes": mem["dloral_worker_allocated"]["peak_bytes"],
        "worker_display": mem["dloral_worker_allocated"]["display"],
        "do_not": mem["do_not"],
    }
    write_json(panel / "PANEL.json", summary)
    write_json(out_dir / "memory_accounting.json", mem)
    html_path = out_dir / "index.html"
    html_path.write_text(
        build_html(panel=panel, figures=figures, mem=mem, summary=summary, selection=selection),
        encoding="utf-8",
    )
    print(json.dumps({"html": str(html_path), "figures": str(figures)}, indent=2))


if __name__ == "__main__":
    main()
