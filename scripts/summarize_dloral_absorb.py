#!/usr/bin/env python3
"""Merge 2x SH-only absorption curves. Per-mode L1 to its own refined.png is not a ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


MODES = ("target_only", "spynet", "geometry")
CROPS = ("building", "vehicles", "trees")
DISCLAIMER = (
    "Each mode is scored against its own refined.png. Those L1/HF numbers "
    "track convergence, not image quality or ranking across modes."
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def _series(curve: list[dict[str, Any]], getter) -> list[tuple[int, float | None]]:
    return [(int(item["step"]), getter(item)) for item in curve]


def _target_l1(item: dict[str, Any]) -> float | None:
    return item.get("target", {}).get("l1_to_refined")


def _target_hf(item: dict[str, Any]) -> float | None:
    return item.get("target", {}).get("hf_l1_to_refined")


def _mean_of(item: dict[str, Any], key: str, field: str) -> float | None:
    return item.get(key, {}).get(field)


def _crop_l1(item: dict[str, Any], name: str) -> float | None:
    return item.get("target", {}).get("crops", {}).get(name, {}).get("l1_to_refined")


def _try_plot(root: Path, payload: dict[str, Any]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], "target.l1_to_refined", "Target L1 to own refined (not a ranking)"),
        (axes[0, 1], "target.hf_l1_to_refined", "Target high-frequency L1 to own refined"),
        (axes[1, 0], "train_views_mean.l1_to_gt_mean", "Train-view L1 to original GT"),
        (axes[1, 1], "heldout_views_mean.l1_to_gt_mean", "Held-out test-view L1 to GT"),
    ]
    for axis, series_key, title in panels:
        for mode in MODES:
            run = payload["runs"].get(mode)
            if not run or not run.get("curve"):
                continue
            xs = [point[0] for point in run["series"][series_key]]
            ys = [point[1] for point in run["series"][series_key]]
            axis.plot(xs, ys, marker="o", label=mode)
        axis.set_title(title)
        axis.set_xlabel("SH-only step")
        axis.grid(True, alpha=0.3)
        axis.legend()
    out = root / "curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def _label(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 28), (16, 16, 16))
    canvas.paste(image, (0, 28))
    draw_canvas = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    draw_canvas.text((8, 6), text, fill=(240, 240, 240), font=font)
    return canvas


def _open_rgb(path: Path) -> Image.Image | None:
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _stack_row(cells: list[Image.Image], gap: int = 8) -> Image.Image:
    height = max(cell.height for cell in cells)
    width = sum(cell.width for cell in cells) + gap * (len(cells) - 1)
    row = Image.new("RGB", (width, height), (8, 8, 8))
    x = 0
    for cell in cells:
        row.paste(cell, (x, 0))
        x += cell.width + gap
    return row


def _stack_col(rows: list[Image.Image], gap: int = 12) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), (8, 8, 8))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap
    return canvas


def _build_crop_montage(root: Path) -> list[str]:
    written: list[str] = []
    for crop in CROPS:
        rows: list[Image.Image] = []
        for mode in MODES:
            step_dir = root / mode / "zoom_2x" / "steps"
            cells = []
            for step, tag in ((0, "step0"), (500, "step500")):
                image = _open_rgb(step_dir / f"{step:04d}" / f"crop_{crop}.png")
                if image is not None:
                    cells.append(_label(image, f"{mode} {tag}"))
            refined = _open_rgb(step_dir / "0000" / f"crop_{crop}_refined.png")
            if refined is not None:
                cells.append(_label(refined, f"{mode} refined"))
            if cells:
                rows.append(_stack_row(cells))
        if not rows:
            continue
        out = root / f"crops_{crop}.png"
        _stack_col(rows).save(out)
        written.append(str(out))
    return written


def _build_heldout_montage(root: Path) -> str | None:
    rows: list[Image.Image] = []
    for mode in MODES:
        step_dir = root / mode / "zoom_2x" / "steps"
        zero = step_dir / "0000"
        last = step_dir / "0500"
        if not zero.is_dir() or not last.is_dir():
            continue
        held0 = sorted(zero.glob("heldout_*.png"))
        cells: list[Image.Image] = []
        for path in held0:
            before = _open_rgb(path)
            after = _open_rgb(last / path.name)
            if before is not None:
                cells.append(_label(before, f"{mode} {path.stem} step0"))
            if after is not None:
                cells.append(_label(after, f"{mode} {path.stem} step500"))
        if cells:
            rows.append(_stack_row(cells))
    if not rows:
        return None
    out = root / "heldout_0_vs_500.png"
    _stack_col(rows).save(out)
    return str(out)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def _markdown(payload: dict[str, Any], artifacts: dict[str, Any]) -> str:
    lines = [
        "# 2× SH-only absorption (500 steps)",
        "",
        DISCLAIMER,
        "",
        "- Checkpoint / ROI / 2× camera / seed / mix_ratio=0.2 are shared.",
        "- Geometry stays frozen; no SR regeneration; no LoD.",
        "- Held-out views are Stage1 test cameras and are never mixed into SH-only sampling.",
        "",
        "## Convergence",
        "",
        "| mode | target L1 0→500 | target HF 0→500 | train GT L1 0→500 | held-out GT L1 0→500 | neighbor GT L1 0→500 |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        run = payload["runs"].get(mode)
        if not run:
            lines.append(f"| {mode} | missing | missing | missing | missing | missing |")
            continue
        d = run["deltas"]
        lines.append(
            "| {mode} | {t0} → {t1} ({dt}) | {h0} → {h1} ({dh}) | {tr0} → {tr1} ({dtr}) | {ho0} → {ho1} ({dho}) | {n0} → {n1} ({dn}) |".format(
                mode=mode,
                t0=_fmt(d["target_l1"]["start"]),
                t1=_fmt(d["target_l1"]["end"]),
                dt=_fmt(d["target_l1"]["delta"]),
                h0=_fmt(d["target_hf"]["start"]),
                h1=_fmt(d["target_hf"]["end"]),
                dh=_fmt(d["target_hf"]["delta"]),
                tr0=_fmt(d["train_l1"]["start"]),
                tr1=_fmt(d["train_l1"]["end"]),
                dtr=_fmt(d["train_l1"]["delta"]),
                ho0=_fmt(d["heldout_l1"]["start"]),
                ho1=_fmt(d["heldout_l1"]["end"]),
                dho=_fmt(d["heldout_l1"]["delta"]),
                n0=_fmt(d["neighbor_l1"]["start"]),
                n1=_fmt(d["neighbor_l1"]["end"]),
                dn=_fmt(d["neighbor_l1"]["delta"]),
            )
        )
    lines.extend(
        [
            "",
            "Negative delta means the quantity decreased (closer to the reference).",
            "",
            "## Local crops vs own refined",
            "",
            "| mode | building L1 0→500 | vehicles L1 0→500 | trees L1 0→500 |",
            "|---|---|---|---|",
        ]
    )
    for mode in MODES:
        run = payload["runs"].get(mode)
        if not run:
            lines.append(f"| {mode} | missing | missing | missing |")
            continue
        crops = run["deltas"]["crops"]
        cells = []
        for crop in CROPS:
            item = crops[crop]
            cells.append(f"{_fmt(item['start'])} → {_fmt(item['end'])} ({_fmt(item['delta'])})")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Artifacts", ""])
    if artifacts.get("curves"):
        lines.append(f"- curves: `{artifacts['curves']}`")
    for path in artifacts.get("crops", []):
        lines.append(f"- crops: `{path}`")
    if artifacts.get("heldout"):
        lines.append(f"- held-out: `{artifacts['heldout']}`")
    if payload.get("view_sequence_identical") is True:
        lines.append("- view sequences are identical across completed modes.")
    elif payload.get("view_sequence_identical") is False:
        lines.append("- WARNING: view sequences differ across modes.")
    lines.append("")
    return "\n".join(lines)


def _endpoints(series: list[tuple[int, float | None]]) -> dict[str, float | None]:
    values = [value for _, value in series if value is not None]
    start = values[0] if values else None
    end = values[-1] if values else None
    return {"start": start, "end": end, "delta": _delta(start, end)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    payload: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "root": str(root),
        "view_sequence_identical": None,
        "runs": {},
    }
    sequences: list[list[dict[str, Any]]] = []
    for mode in MODES:
        curve_path = root / mode / "zoom_2x" / "absorption_curve.json"
        seq_path = root / mode / "zoom_2x" / "view_sequence.json"
        if not curve_path.is_file():
            continue
        curve = _load_json(curve_path)
        sequence = _load_json(seq_path) if seq_path.is_file() else {}
        view_sequence = sequence.get("view_sequence", [])
        if view_sequence:
            sequences.append(view_sequence)
        series = {
            "target.l1_to_refined": _series(curve, _target_l1),
            "target.hf_l1_to_refined": _series(curve, _target_hf),
            "train_views_mean.l1_to_gt_mean": _series(
                curve, lambda item: _mean_of(item, "train_views_mean", "l1_to_gt_mean")
            ),
            "heldout_views_mean.l1_to_gt_mean": _series(
                curve, lambda item: _mean_of(item, "heldout_views_mean", "l1_to_gt_mean")
            ),
            "neighbors_mean.l1_to_gt_mean": _series(
                curve, lambda item: _mean_of(item, "neighbors_mean", "l1_to_gt_mean")
            ),
        }
        crop_series = {name: _series(curve, lambda item, crop=name: _crop_l1(item, crop)) for name in CROPS}
        payload["runs"][mode] = {
            "curve_path": str(curve_path),
            "curve": curve,
            "series": series,
            "deltas": {
                "target_l1": _endpoints(series["target.l1_to_refined"]),
                "target_hf": _endpoints(series["target.hf_l1_to_refined"]),
                "train_l1": _endpoints(series["train_views_mean.l1_to_gt_mean"]),
                "heldout_l1": _endpoints(series["heldout_views_mean.l1_to_gt_mean"]),
                "neighbor_l1": _endpoints(series["neighbors_mean.l1_to_gt_mean"]),
                "crops": {name: _endpoints(crop_series[name]) for name in CROPS},
            },
        }

    if len(sequences) >= 2:
        payload["view_sequence_identical"] = all(item == sequences[0] for item in sequences[1:])

    artifacts = {
        "curves": _try_plot(root, payload),
        "crops": _build_crop_montage(root),
        "heldout": _build_heldout_montage(root),
    }
    payload["artifacts"] = artifacts
    summary_path = root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    markdown_path = root / "summary.md"
    markdown_path.write_text(_markdown(payload, artifacts), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {markdown_path}")
    if artifacts["curves"]:
        print(f"Wrote {artifacts['curves']}")
    for path in artifacts["crops"]:
        print(f"Wrote {path}")
    if artifacts["heldout"]:
        print(f"Wrote {artifacts['heldout']}")


if __name__ == "__main__":
    main()
