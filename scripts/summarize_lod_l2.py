#!/usr/bin/env python3
"""Summarize 4x L2 geometry absorption against the frozen L1.

SpyNet 2x is a previous-scale baseline only. Do not rank alignment modes here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _row(curve, step: int):
    for item in curve:
        if int(item["step"]) == step:
            return item
    return None


def _rel(before, after):
    if before in (None, 0):
        return None
    return (after - before) / before


def _mean_excess(payload: dict) -> dict:
    rows = list((payload.get("neighbors") or {}).values())
    if not rows:
        return {}

    def pick(*keys):
        values = []
        for row in rows:
            value = row
            for key in keys:
                value = (value or {}).get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(value)
        return None if not values else sum(values) / len(values)

    return {
        "direct_change_rgb": pick("neighbor_vs_l0_direct", "rgb_l1"),
        "excess_hf": pick("l0_depth", "excess_over_floor", "hf_l1"),
        "matched_hf": pick("l0_depth", "l1_rgb", "hf_l1"),
        "floor_hf": pick("l0_depth", "floor_l0_rgb", "hf_l1"),
        "newly_occluded": pick("trained_depth", "frac_newly_occluded"),
        "coverage": pick("l0_depth", "coverage"),
    }


def _eval_curve(curve) -> list[dict]:
    rows = []
    for item in curve:
        active = item.get("active") or {}
        parent = item.get("parent_2x") or {}
        rows.append({
            "step": int(item["step"]),
            "n": item.get("n_l2", active.get("n")),
            "rgb": item["target"]["l1_to_refined"],
            "hf": item["target"]["hf_l1_to_refined"],
            "train_gt_1x": (item.get("train_views_mean") or {}).get("l1_to_gt_mean"),
            "parent_2x_rgb": parent.get("l1_vs_frozen_l1"),
            "l2_weight_at_2x": parent.get("l2_weight_mean"),
            "opacity_mean": active.get("opacity_mean"),
            "frac_opacity_lt_0.05": active.get("frac_opacity_lt_0.05"),
        })
    return rows


def _sweep_l2_at_2x(cross: dict):
    for row in (cross.get("scale_sweep") or {}).get("rows") or []:
        if abs(float(row.get("factor", 0.0)) - 2.0) < 1e-6:
            return (row.get("weights") or {}).get("level_2")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--spynet_2x", type=str, default="")
    args = parser.parse_args()
    root = Path(args.root)
    curve_path = root / "absorption_curve.json"
    if not curve_path.is_file():
        raise FileNotFoundError(curve_path)
    data = _load(curve_path)
    curve = data["curve"]
    s0 = _row(curve, 0) or curve[0]
    s50 = _row(curve, 50)
    s500 = _row(curve, 500) or curve[-1]
    correspondence = _load(root / "correspondence.json") if (root / "correspondence.json").is_file() else {}
    cross = _load(root / "cross_view.json") if (root / "cross_view.json").is_file() else {}
    geometry_sup = _load(root / "geometry.json") if (root / "geometry.json").is_file() else {}
    densify_log = data.get("densify_log")
    if densify_log is None and (root / "densify_log.json").is_file():
        densify_log = _load(root / "densify_log.json")
    eval_curve = _eval_curve(curve)
    n50 = None if s50 is None else s50.get("n_l2")
    n_final = s500.get("n_l2")
    budget_full_by_50 = n50 is not None and n_final and n50 >= 0.99 * float(n_final)
    rgb_worse_at_50 = (
        s50 is not None
        and s50["target"]["l1_to_refined"] > s0["target"]["l1_to_refined"]
    )
    rgb_recovered = s500["target"]["l1_to_refined"] < s0["target"]["l1_to_refined"]
    parent_2x = s500.get("parent_2x") or {}
    payload = {
        "n_l2": data.get("n_l2"),
        "alignment": geometry_sup.get("alignment"),
        "best_coverage": geometry_sup.get("best_coverage"),
        "l1_0": s0["target"]["l1_to_refined"],
        "hf_0": s0["target"]["hf_l1_to_refined"],
        "l1_50": None if s50 is None else s50["target"]["l1_to_refined"],
        "hf_50": None if s50 is None else s50["target"]["hf_l1_to_refined"],
        "l1_final": s500["target"]["l1_to_refined"],
        "hf_final": s500["target"]["hf_l1_to_refined"],
        "l1_rel": _rel(s0["target"]["l1_to_refined"], s500["target"]["l1_to_refined"]),
        "hf_rel": _rel(s0["target"]["hf_l1_to_refined"], s500["target"]["hf_l1_to_refined"]),
        "train_gt_0": (s0.get("train_views_mean") or {}).get("l1_to_gt_mean"),
        "train_gt_final": (s500.get("train_views_mean") or {}).get("l1_to_gt_mean"),
        "parent_2x_final": parent_2x,
        "l2_weight_at_2x": parent_2x.get("l2_weight_mean") or _sweep_l2_at_2x(cross),
        "scale_sweep_max_jump": (cross.get("scale_sweep") or {}).get("max_mean_rgb_jump"),
        "correspondence": _mean_excess(correspondence),
        "eval_curve": eval_curve,
        "densify_log_len": 0 if densify_log is None else len(densify_log),
        "acceptance": {
            "absorption": {
                "rgb_rel": _rel(s0["target"]["l1_to_refined"], s500["target"]["l1_to_refined"]),
                "hf_rel": _rel(s0["target"]["hf_l1_to_refined"], s500["target"]["hf_l1_to_refined"]),
            },
            "early_densify": {
                "budget_full_by_step_50": budget_full_by_50,
                "rgb_worse_at_step_50": rgb_worse_at_50,
                "rgb_recovered_by_final": rgb_recovered,
                "n_50": n50,
                "n_final": n_final,
                "opacity_at_eval_steps": [
                    {"step": row["step"], "opacity_mean": row["opacity_mean"], "frac_lt_0.05": row["frac_opacity_lt_0.05"]}
                    for row in eval_curve
                ],
                "note": (
                    "Step-50 degradation can be tied to filling the point budget only if "
                    "n/opacity/loss are read together. A 0/50/500 RGB triple is not enough."
                ),
            },
            "frozen_params_vs_parent_scale": {
                "train_gt_1x_unchanged": (
                    abs(float((s0.get("train_views_mean") or {}).get("l1_to_gt_mean") or 0.0)
                        - float((s500.get("train_views_mean") or {}).get("l1_to_gt_mean") or 0.0)) < 1e-8
                ),
                "l2_weight_at_2x": parent_2x.get("l2_weight_mean") or _sweep_l2_at_2x(cross),
                "rgb_l1_at_2x_vs_frozen_l1": parent_2x.get("l1_vs_frozen_l1"),
                "label": "l2_contribution_at_parent_scale",
                "not": "leak",
            },
            "correspondence": {
                "supports": "stability_of_co_visible_region",
                "does_not_support": ["photorealism", "alignment_mode_ranking"],
                "do_not_compare_excess_hf_across_scales": True,
            },
        },
        "spynet_2x_note": "SpyNet 2x remains the previous-scale baseline; this run does not claim geometry > SpyNet.",
    }
    if args.spynet_2x:
        spy = Path(args.spynet_2x)
        if (spy / "absorption_curve.json").is_file():
            spy_data = _load(spy / "absorption_curve.json")
            spy_curve = spy_data["curve"]
            spy_final = _row(spy_curve, 500) or spy_curve[-1]
            payload["spynet_2x"] = {
                "l1_final": spy_final["target"]["l1_to_refined"],
                "hf_final": spy_final["target"]["hf_l1_to_refined"],
                "train_gt_final": (spy_final.get("train_views_mean") or {}).get("l1_to_gt_mean"),
            }
    print(json.dumps(payload, indent=2))
    out = root / "lod_l2_summary.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
