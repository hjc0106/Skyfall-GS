#!/usr/bin/env python3
"""Compare target_only / SpyNet / geometry L1 2x absorption on the joint RaDe path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODES = ("target_only", "spynet", "geometry")


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _target(curve, step: int):
    for item in curve:
        if int(item["step"]) == step:
            return item
    return None


def _delta(a, b):
    if a is None or b is None:
        return None
    return b - a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    payload = {"runs": {}, "disclaimer": "Each mode is scored against its own refined.png."}
    print(f"{'mode':<12} {'L1 0':>8} {'L1 500':>8} {'dL1%':>7} {'HF 0':>8} {'HF 500':>8} {'dHF%':>7} {'veh%':>7} {'n_l1':>6} {'jump':>8} {'off_p50':>8}")
    for mode in MODES:
        curve_path = root / mode / "absorption_curve.json"
        cross_path = root / mode / "cross_view.json"
        if not curve_path.is_file():
            continue
        data = _load(curve_path)
        curve = data["curve"]
        s0 = _target(curve, 0)
        s1 = _target(curve, 500) or curve[-1]
        l0 = s0["target"]["l1_to_refined"]
        l1 = s1["target"]["l1_to_refined"]
        h0 = s0["target"]["hf_l1_to_refined"]
        h1 = s1["target"]["hf_l1_to_refined"]
        v0 = s0["target"]["crops"]["vehicles"]["l1_to_refined"]
        v1 = s1["target"]["crops"]["vehicles"]["l1_to_refined"]
        cross = _load(cross_path) if cross_path.is_file() else {}
        geom = cross.get("geometry") or {}
        jump = (cross.get("scale_sweep") or {}).get("max_mean_rgb_jump")
        offset = ((geom.get("offset_to_parent") or {}).get("p50"))
        payload["runs"][mode] = {
            "n_l1": data.get("n_l1"),
            "l1_0": l0,
            "l1_500": l1,
            "hf_0": h0,
            "hf_500": h1,
            "l1_rel": _delta(l0, l1) / l0 if l0 else None,
            "hf_rel": _delta(h0, h1) / h0 if h0 else None,
            "vehicles_rel": _delta(v0, v1) / v0 if v0 else None,
            "train_gt_0": s0.get("train_views_mean", {}).get("l1_to_gt_mean"),
            "train_gt_500": s1.get("train_views_mean", {}).get("l1_to_gt_mean"),
            "held_gt_0": s0.get("heldout_views_mean", {}).get("l1_to_gt_mean"),
            "held_gt_500": s1.get("heldout_views_mean", {}).get("l1_to_gt_mean"),
            "max_mean_rgb_jump": jump,
            "offset_to_parent_p50": offset,
            "neighbors": (cross.get("views") or {}).get("neighbors"),
            "test_2x": (cross.get("views") or {}).get("test_2x"),
        }
        rec = payload["runs"][mode]
        print(
            f"{mode:<12} {l0:8.4f} {l1:8.4f} {100*rec['l1_rel']:6.1f}% "
            f"{h0:8.4f} {h1:8.4f} {100*rec['hf_rel']:6.1f}% "
            f"{100*rec['vehicles_rel']:6.1f}% {rec['n_l1'] or 0:6d} "
            f"{(jump or 0):8.4f} {(offset or 0):8.4f}"
        )
    out = root / "lod_l1_summary.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
