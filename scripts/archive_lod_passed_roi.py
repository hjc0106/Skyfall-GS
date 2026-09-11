#!/usr/bin/env python3
"""Archive the passed JAX_068 Y-building ROI as a closed 2x->4x example.

Writes a manifest and relative symlinks. Does not copy multi-hundred-MB checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT_DEFAULT = Path(__file__).resolve().parents[1]
PASSED_ROI = {
    "name": "jax068_ybuilding",
    "scene": "JAX_068",
    "view_index": 0,
    "roi": {"center_x": 0.592, "center_y": 0.53, "width": 0.1, "height": 0.1},
    "status": "passed_small_scope",
    "closed_loop": "per-scale supervision -> add detail level -> 3D absorption",
    "scope": ["supervision_absorption", "old_scale_influence", "co_visible_stability"],
    "out_of_scope": ["generated_detail_photorealism", "geometry_better_than_spynet"],
}


def _rel_symlink(archive: Path, src: Path, dest_name: str) -> str | None:
    if not src.exists():
        return None
    dest = archive / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(os.path.relpath(src.resolve(), dest.parent))
    return dest_name


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _eval_rows(root: Path) -> list[dict]:
    rows = []
    for metrics in sorted(root.glob("steps/*/metrics.json")):
        data = _load(metrics)
        rows.append({
            "step": data["step"],
            "n": data.get("n_l2", data.get("n_l1")),
            "rgb": data["target"]["l1_to_refined"],
            "hf": data["target"]["hf_l1_to_refined"],
            "train_gt_1x": (data.get("train_views_mean") or {}).get("l1_to_gt_mean"),
            "parent_2x_rgb": (data.get("parent_2x") or {}).get("l1_vs_frozen_l1"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, default=str(ROOT_DEFAULT))
    parser.add_argument("--l1_dir", type=str, default="skyfall-gs_exp/lod_l1_absorb_2x/geometry")
    parser.add_argument("--l2_dir", type=str, default="skyfall-gs_exp/lod_l2_geometry_4x")
    parser.add_argument("--archive", type=str, default="skyfall-gs_exp/lod_passed_roi_jax068_ybuilding")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    archive = (repo / args.archive).resolve()
    l1 = (repo / args.l1_dir).resolve()
    l2 = (repo / args.l2_dir).resolve()
    archive.mkdir(parents=True, exist_ok=True)

    links = {}
    for src, name in (
        (l2 / "refined.png", "supervision/l2_4x_refined.png"),
        (l2 / "render_input.png", "supervision/l2_4x_render_input.png"),
        (l2 / "neighbor.png", "supervision/l2_4x_neighbor.png"),
        (l2 / "neighbor_warped.png", "supervision/l2_4x_neighbor_warped.png"),
        (l2 / "flow_valid.png", "supervision/l2_4x_flow_valid.png"),
        (l2 / "geometry.json", "supervision/l2_4x_geometry.json"),
        (l2 / "parent_2x_frozen.png", "renders/parent_2x_frozen.png"),
        (l2 / "render_before.png", "renders/l2_step0.png"),
        (l2 / "steps/0050/target.png", "renders/l2_step50.png"),
        (l2 / "steps/0500/target.png", "renders/l2_step500.png"),
        (l2 / "absorption_curve.json", "curves/l2_absorption_curve.json"),
        (l2 / "correspondence.json", "diagnostics/l2_correspondence.json"),
        (l2 / "cross_view.json", "diagnostics/l2_cross_view.json"),
        (l2 / "lod_l2_summary.json", "diagnostics/lod_l2_summary.json"),
        (l2 / "densify_probe/densify_log.json", "curves/l2_densify_probe.json"),
        (l2 / "densify_probe/steps/0050/metrics.json", "curves/l2_densify_probe_step50.json"),
        (l1 / "l1_final.lod.pt", "checkpoints/l1_final.lod.pt"),
        (l1 / "l1_final.lod.pt.appearance.pt", "checkpoints/l1_final.lod.pt.appearance.pt"),
        (l2 / "l2_final.lod.pt", "checkpoints/l2_final.lod.pt"),
        (l2 / "l2_final.lod.pt.appearance.pt", "checkpoints/l2_final.lod.pt.appearance.pt"),
        (l1 / "absorption_curve.json", "curves/l1_absorption_curve.json"),
        (l1 / "correspondence.json", "diagnostics/l1_correspondence.json"),
        (Path(repo) / "skyfall-gs_exp/zoom_gen_dloral_align_2048/geometry/refined.png",
         "supervision/l1_2x_refined.png"),
    ):
        linked = _rel_symlink(archive, src, name)
        if linked:
            links[name] = str(src)

    l2_eval = _eval_rows(l2)
    l1_eval = _eval_rows(l1)
    summary = _load(l2 / "lod_l2_summary.json") if (l2 / "lod_l2_summary.json").is_file() else {}
    sweep = (_load(l2 / "cross_view.json").get("scale_sweep") if (l2 / "cross_view.json").is_file() else {}) or {}
    video_json = archive / "video" / "zoom_1p25_to_4.json"
    video_meta = _load(video_json) if video_json.is_file() else {}
    densify_probe = {}
    densify_path = l2 / "densify_probe" / "densify_log.json"
    if densify_path.is_file():
        densify_probe = {
            "source": "seed-matched 50-step rerun, does not replace the 500-step checkpoint",
            "log": [row for row in _load(densify_path)],
        }
    manifest = {
        **PASSED_ROI,
        "links": links,
        "l1_eval": l1_eval,
        "l2_eval": l2_eval,
        "densify_probe": densify_probe,
        "acceptance": {
            "absorption": {
                "rgb_0": summary.get("l1_0"),
                "rgb_50": summary.get("l1_50"),
                "rgb_500": summary.get("l1_final"),
                "hf_0": summary.get("hf_0"),
                "hf_50": summary.get("hf_50"),
                "hf_500": summary.get("hf_final"),
                "note": (
                    "500-step RGB -46% / HF -15% support L2 absorbing 4x supervision. "
                    "Record early perturbation as: rapid introduction of high-opacity new "
                    "Gaussians (mean opacity ~0.984) coincides with the early fit drop; "
                    "error falls after the point budget is reached and optimization continues. "
                    "Opacity logits still pass through LoD weights, filtering, and projection, "
                    "so they are not per-pixel occlusion. "
                    "'New capacity arrived faster than fitting' remains a mechanism hypothesis. "
                    "Do not change densify policy or add geometry constraints: 500 steps recover."
                ),
            },
            "frozen_params_vs_parent_scale": {
                "train_gt_1x": summary.get("train_gt_final"),
                "l2_weight_at_2x": next(
                    ((row.get("weights") or {}).get("level_2") for row in sweep.get("rows") or []
                     if abs(float(row.get("factor", 0)) - 2.0) < 1e-6),
                    None,
                ),
                "rgb_l1_at_2x_vs_frozen_l1": (summary.get("parent_2x_final") or {}).get("l1_vs_frozen_l1"),
                "label": "l2_contribution_at_parent_scale",
                "not": "leak",
                "keep_as_fixed_metric": True,
            },
            "correspondence": {
                **(summary.get("correspondence") or {}),
                "supports": "stability_of_co_visible_region",
                "does_not_support": ["photorealism", "alignment_mode_ranking"],
                "do_not_compare_to_2x_excess_hf": True,
            },
            "continuous_zoom": {
                "max_mean_rgb_jump": sweep.get("max_mean_rgb_jump"),
                "video_max_mean_rgb_jump": video_meta.get("max_mean_rgb_jump"),
                "video": "video/zoom_1p25_to_4.mp4",
                "do_not_compare_sparse_and_dense_jumps": True,
                "check_manually": ["building edges", "vehicle silhouettes"],
                "note": "0.0046 on 36 frames is not comparable to 0.0087 on the coarser 0.5-step sweep.",
            },
        },
        "next": [
            "This ROI is the passed sample. Scope is absorption, old-scale influence, and co-visible stability only.",
            "Next ROI: larger parallax / stronger occlusion, still 2x->4x.",
            "Gate the new ROI on correspondence coverage and occlusion masks before ranking alignments.",
            "SpyNet vs geometry must share one frozen upstream; do not mix L1 checkpoints into the alignment comparison.",
            "Do not go to 8x yet.",
        ],
    }
    out = archive / "ACCEPTANCE.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"archive": str(archive), "n_links": len(links), "manifest": str(out)}, indent=2))


if __name__ == "__main__":
    main()
