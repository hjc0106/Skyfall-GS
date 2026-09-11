#!/usr/bin/env python3
"""Print whether neighbor-view L1 edits correspond after depth warp, or just change more."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODES = ("target_only", "spynet", "geometry")


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values):
    values = [item for item in values if item is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _neighbor_rows(payload: dict) -> list[dict]:
    return list((payload.get("neighbors") or {}).values())


def _pick(rows: list[dict], *keys):
    values = []
    for row in rows:
        value = row
        for key in keys:
            value = (value or {}).get(key) if isinstance(value, dict) else None
        values.append(value)
    return _mean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    print(
        f"{'mode':<12} {'chgRGB':>8} {'floorHF':>8} {'matchHF':>8} {'exHF':>8} "
        f"{'unexp':>7} {'vehEx':>8} {'occEx':>8} {'shExHF':>8} {'outConc':>8} {'newOcc':>8}"
    )
    summary = {"modes": {}, "read": "excess HF over the L0 warp floor; unexplained_ratio near 0 means neighbor edits match warped target edits."}
    for mode in MODES:
        path = root / mode / "correspondence.json"
        if not path.is_file():
            continue
        data = _load(path)
        rows = _neighbor_rows(data)
        rec = {
            "direct_change_rgb": _pick(rows, "neighbor_vs_l0_direct", "rgb_l1"),
            "floor_hf": _pick(rows, "l0_depth", "floor_l0_rgb", "hf_l1"),
            "matched_hf": _pick(rows, "l0_depth", "l1_rgb", "hf_l1"),
            "excess_hf": _pick(rows, "l0_depth", "excess_over_floor", "hf_l1"),
            "excess_rgb": _pick(rows, "l0_depth", "excess_over_floor", "rgb_l1"),
            "unexplained_ratio": _pick(rows, "l0_depth", "change_agreement", "unexplained_ratio"),
            "vehicles_excess_hf": _pick(rows, "l0_depth", "excess_over_floor", "crops", "vehicles", "hf_l1"),
            "building_excess_hf": _pick(rows, "l0_depth", "excess_over_floor", "crops", "building", "hf_l1"),
            "occlusion_excess_hf": _pick(rows, "l0_depth", "excess_over_floor", "occlusion_boundary", "hf_l1"),
            "shared_emb_excess_hf": _pick(rows, "l0_depth", "shared_target_embedding", "excess_over_floor", "hf_l1"),
            "trained_matched_hf": _pick(rows, "trained_depth", "l1_rgb", "hf_l1"),
            "newly_occluded": _pick(rows, "trained_depth", "frac_newly_occluded"),
            "outlier_concentration": _pick(rows, "outliers_on_l0depth_l1_residual", "scale_or_offset_gt_4x", "concentration_vs_all_l1"),
            "l0_coverage": _pick(rows, "l0_depth", "coverage"),
            "trained_coverage": _pick(rows, "trained_depth", "coverage"),
        }
        summary["modes"][mode] = rec
        print(
            f"{mode:<12} {rec['direct_change_rgb'] or 0:8.4f} {rec['floor_hf'] or 0:8.4f} "
            f"{rec['matched_hf'] or 0:8.4f} {rec['excess_hf'] or 0:8.4f} "
            f"{rec['unexplained_ratio'] or 0:7.3f} {rec['vehicles_excess_hf'] or 0:8.4f} "
            f"{rec['occlusion_excess_hf'] or 0:8.4f} {rec['shared_emb_excess_hf'] or 0:8.4f} "
            f"{rec['outlier_concentration'] or 0:8.3f} {rec['newly_occluded'] or 0:8.4f}"
        )
    out = root / "lod_l1_correspondence_summary.json"
    existing = _load(out) if out.is_file() else {}
    existing["averages"] = summary
    with out.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
