#!/usr/bin/env python3
"""Build full-scene SR supervision for the LoD trainer.

Without --flowedit_manifests, collect the checkpoint's real TRAIN views.
With FlowEdit manifests, collect the repaired native views; --include_real_replay
also retains the original TRAIN views for replay, without generating SR targets
for them. --target_view_ids partitions target generation only: replay/context
views and the geometry-neighbor pool remain complete in every shard.

In progressive mode, render each zoomed tile from the current parent model for
DLoRAL input and retain the corresponding native-image crop as the LR anchor.
--previous_supervision carries completed lower-level targets forward unchanged.

For the complete current training recipe and portable resource configuration:
    python scripts/train_widefe_gszoom.py --config configs/widefe_gszoom.example.json --plan-only

Run this script with --help for the individual preparation-stage arguments.
Model paths and Python environments must be supplied explicitly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refinement.scene_zoom import (  # noqa: E402
    SceneZoomConfig,
    run_scene_zoom,
)


def parse_factors(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("--zoom_factors must contain at least one value.")
    if any(value <= 1.0 for value in values):
        raise ValueError("every zoom factor must be > 1.0.")
    if any(current <= previous for previous, current in zip(values, values[1:])):
        raise ValueError("--zoom_factors must be strictly increasing.")
    for value in values:
        if abs(value - round(value)) > 1e-9:
            raise ValueError(f"zoom factor {value:g} is not an integer tile-grid size.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare whole-scene zoom SR supervision for the RaDe LoD trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start_checkpoint", type=str, required=True,
                        help="Path to the source checkpoint .pth (cfg_args + PLY live in its directory).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Fresh directory receiving supervision.json plus artifacts.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse verified completed target records for the same generation run.")
    parser.add_argument("--source_path", type=str, default=None,
                        help="Dataset root override; wins over the checkpoint cfg_args.")
    parser.add_argument("--flowedit_manifests", type=str, nargs="*", default=[],
                        help="Stage 2: flowedit_views.json files (kind skyfall_flowedit_views).")
    parser.add_argument("--real_supervision", type=str, default=None,
                        help="Stage 2: Stage 1 supervision.json carried over verbatim.")
    parser.add_argument("--zoom_factors", type=str, default="2,4",
                        help="Integer focal-zoom grid factors, e.g. 2,4.")
    parser.add_argument("--resolution", type=int, default=0,
                        help="Base raster long side; 0 keeps each input's native raster.")
    parser.add_argument("--max_views", type=int, default=0,
                        help="Cap on new views processed; 0 keeps all (smoke testing only).")
    parser.add_argument("--target_view_ids", type=str, nargs="+", default=[],
                        help="Sharded generation: generate new SR targets only for these "
                             "collected view ids; the full view/context/neighbor pool is "
                             "still collected. Unknown or replay-only ids are refused.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vlm_python", type=str, required=True,
                        help="Python interpreter of the qwen3vl environment.")
    parser.add_argument("--vlm_model_path", type=str, required=True,
                        help="Local Qwen3-VL model directory.")
    parser.add_argument("--vlm_device", type=str, default="cuda:0")
    parser.add_argument("--vlm_max_new_tokens", type=int, default=768)
    parser.add_argument("--vlm_max_image_size", type=int, default=1024)
    parser.add_argument("--dloral_python", type=str, required=True,
                        help="Python interpreter of the dloral environment.")
    parser.add_argument("--dloral_root", type=str, default=None,
                        help="Pinned DLoRAL repository root (defaults to submodules/DLoRAL).")
    parser.add_argument("--dloral_sd_path", type=str, required=True,
                        help="Local Stable Diffusion 2.1 base directory consumed by DLoRAL.")
    parser.add_argument("--dloral_ckpt", type=str, required=True,
                        help="Official improved DLoRAL checkpoint file.")
    parser.add_argument("--dloral_spynet", type=str, required=True,
                        help="Local SpyNet checkpoint file.")
    parser.add_argument("--dloral_device", type=str, default="cuda:0")
    parser.add_argument("--dloral_stages", type=int, default=1)
    parser.add_argument("--dloral_process_size", type=int, default=512)
    parser.add_argument("--dloral_align_method", type=str, default="adain",
                        choices=["adain", "wavelet", "nofix"])
    parser.add_argument("--dloral_alignment", type=str, default="geometry",
                        choices=["geometry", "spynet", "target_only"],
                        help="geometry = external depth flows; spynet = native warping; "
                             "target_only = explicit single-view DLoRAL.")
    parser.add_argument("--dloral_max_roundtrip_error_px", type=float, default=None)
    parser.add_argument("--geometry_neighbor_count", type=int, default=1,
                        help="Spatial neighbors per tile for DLoRAL's dual-view path; "
                             "0 disables the geometry phase entirely.")
    parser.add_argument("--geometry_pool_size", type=int, default=8)
    parser.add_argument("--min_target_confidence", type=float, default=0.05)
    parser.add_argument("--min_in_frustum", type=float, default=1e-3)
    parser.add_argument("--depth_abs_tolerance", type=float, default=5.0)
    parser.add_argument("--depth_rel_tolerance", type=float, default=1e-5)
    parser.add_argument("--depth_scene_scale", type=float, default=-1.0,
                        help="Occlusion cap in scene units; <0 auto, 0 disables, >0 explicit.")
    parser.add_argument("--alpha_threshold", type=float, default=1e-4)
    parser.add_argument("--min_depth", type=float, default=1e-4)
    parser.add_argument("--min_reprojection_coverage", type=float, default=0.0,
                        help="Coverage below this is reported per tile; tiles still keep "
                             "their geometry neighbor (no silent target_only).")
    parser.add_argument("--geometry_batch_views", type=int, default=4,
                        help="Views per scene-load cycle between generative phases.")
    parser.add_argument("--prompt_cache_dir", type=str, default=None)
    parser.add_argument("--generation_cache_dir", type=str, default=None)
    parser.add_argument("--no_generation_cache", action="store_true")
    parser.add_argument("--progressive", action="store_true",
                        help="Recursive supervision: render G(t-1) instead of the G0 source "
                             "model at the next-level low-res raster (one new cumulative "
                             "zoom per invocation).")
    parser.add_argument("--parent_lod_checkpoint", type=str, default=None,
                        help="Progressive only: trained G(t-1) LoD checkpoint (.lod.pt) "
                             "imported on top of the frozen G0 bundle for renders.")
    parser.add_argument("--previous_supervision", type=str, default=None,
                        help="Progressive chain: prior-level supervision.json carried "
                             "verbatim into the new manifest without regeneration.")
    parser.add_argument("--local_flowedit_config", type=str, default=None,
                        help="JSON config for C's batch LocalizedFlowEdit (weights_path, "
                             "prompts, sampler, optional view_masks per view).")
    parser.add_argument("--include_real_replay", action="store_true",
                        help="Stage 2 only: additionally collect the checkpoint's native TRAIN "
                             "views as replay-only manifest entries (source_stage real_replay). "
                             "They feed base replay supervision in the LoD trainer but never "
                             "receive SR targets and never count against --max_views.")
    parser.add_argument("--step_scale", type=float, default=2.0,
                        help="Progressive per-level SR factor (SR input raster = "
                             "base_raster/step_scale).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = SceneZoomConfig(
        start_checkpoint=os.path.abspath(args.start_checkpoint),
        output_dir=os.path.abspath(args.output_dir),
        resume=args.resume,
        zoom_factors=parse_factors(args.zoom_factors),
        resolution=args.resolution,
        max_views=args.max_views,
        target_view_ids=[str(view_id) for view_id in args.target_view_ids],
        seed=args.seed,
        source_path=os.path.abspath(args.source_path) if args.source_path else None,
        flowedit_manifests=[os.path.abspath(path) for path in args.flowedit_manifests],
        real_supervision=os.path.abspath(args.real_supervision) if args.real_supervision else None,
        vlm_model_path=args.vlm_model_path,
        vlm_python=args.vlm_python,
        vlm_device=args.vlm_device,
        vlm_max_new_tokens=args.vlm_max_new_tokens,
        vlm_max_image_size=args.vlm_max_image_size,
        dloral_root=args.dloral_root,
        dloral_sd_path=os.path.abspath(args.dloral_sd_path),
        dloral_ckpt=os.path.abspath(args.dloral_ckpt),
        dloral_spynet=os.path.abspath(args.dloral_spynet),
        dloral_python=args.dloral_python,
        dloral_device=args.dloral_device,
        dloral_stages=args.dloral_stages,
        dloral_process_size=args.dloral_process_size,
        dloral_align_method=args.dloral_align_method,
        progressive=args.progressive,
        parent_lod_checkpoint=(
            os.path.abspath(args.parent_lod_checkpoint) if args.parent_lod_checkpoint else None
        ),
        previous_supervision=(
            os.path.abspath(args.previous_supervision) if args.previous_supervision else None
        ),
        local_flowedit_config=(
            os.path.abspath(args.local_flowedit_config) if args.local_flowedit_config else None
        ),
        step_scale=float(args.step_scale),
        include_real_replay=args.include_real_replay,
        dloral_max_roundtrip_error_px=args.dloral_max_roundtrip_error_px,
        min_reprojection_coverage=args.min_reprojection_coverage,
        geometry_neighbor_count=args.geometry_neighbor_count,
        geometry_pool_size=args.geometry_pool_size,
        min_target_confidence=args.min_target_confidence,
        min_in_frustum=args.min_in_frustum,
        depth_abs_tolerance=args.depth_abs_tolerance,
        depth_rel_tolerance=args.depth_rel_tolerance,
        depth_scene_scale=args.depth_scene_scale,
        alpha_threshold=args.alpha_threshold,
        min_depth=args.min_depth,
        geometry_batch_views=args.geometry_batch_views,
        prompt_cache_dir=os.path.abspath(args.prompt_cache_dir) if args.prompt_cache_dir else None,
        generation_cache_dir=os.path.abspath(args.generation_cache_dir) if args.generation_cache_dir else None,
        no_generation_cache=args.no_generation_cache,
    )
    summary = run_scene_zoom(cfg)
    print(
        f"[scene-zoom] done: {summary['samples_generated']} SR target(s) across "
        f"{summary['views_pending']} view(s); manifest: {summary['supervision_manifest']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
