from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from prf.cameras import load_training_views
from prf.config import PRFConfig
from prf.diagnostics import (
    compare_field_statistics,
    compare_observation_archives,
    diagnose_field_archive,
    diagnose_rebuilt_geometry,
    write_diagnostics,
    write_region_spot_checks,
)
from prf.export import export_field
from prf.io_gs import load_gaussians_ply
from prf.manifest import build_experiment_manifest, canonical_path, config_dict, write_manifest
from prf.pipeline import compute_from_paths, recompute_observation_from_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a 3D Physical Resolution Field from a metric 3DGS PLY and training cameras."
    )
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, required=True, help="Original training transforms JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opengl-c2w", action="store_true", help="Treat transform_matrix as OpenGL C2W")
    parser.add_argument("--max-anchors", type=int, default=8000)
    parser.add_argument("--anchor-voxel-m", type=float, default=4.0)
    parser.add_argument("--min-pair-angle-deg", type=float, default=12.0)
    parser.add_argument(
        "--visibility",
        choices=("frustum", "gs-depth"),
        default="frustum",
        help="frustum: image bounds only. gs-depth: training-view expected depth occlusion.",
    )
    parser.add_argument("--reuse-patches", type=Path, help="Frozen schema-v2 NPZ; recompute R_obs only")
    parser.add_argument("--compare-to", type=Path, help="Baseline NPZ to compare against after recompute")
    parser.add_argument("--depth-cache", type=Path, help="Directory for per-view depth/alpha NPZ cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnose-only", action="store_true", help="Write patch geometry diagnostics and exit")
    parser.add_argument("--skip-rebuild-geometry", action="store_true")
    parser.add_argument("--spot-check", action="store_true", help="Write PLY + training-image overlays for region samples")
    parser.add_argument("--compare-stats-to", type=Path, help="Baseline NPZ for statistical geometry comparison")
    return parser


def _cfg(args: argparse.Namespace) -> PRFConfig:
    return PRFConfig(
        max_anchors=args.max_anchors,
        anchor_voxel_m=args.anchor_voxel_m,
        min_pair_angle_deg=args.min_pair_angle_deg,
    )


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    args = build_parser().parse_args()
    cfg = _cfg(args)
    args.output.mkdir(parents=True, exist_ok=True)
    ply_path = canonical_path(args.ply)
    transforms_path = canonical_path(args.transforms)

    if args.diagnose_only:
        archive = _load_archive(args.reuse_patches) if args.reuse_patches else None
        if archive is None and (args.output / "physical_resolution_field.npz").is_file():
            archive = _load_archive(args.output / "physical_resolution_field.npz")
        if archive is None:
            raise SystemExit("--diagnose-only needs --reuse-patches or an existing field NPZ in --output")
        payload = {"field": diagnose_field_archive(archive)}
        if not args.skip_rebuild_geometry:
            payload["rebuilt"] = diagnose_rebuilt_geometry(load_gaussians_ply(ply_path), cfg)
        write_diagnostics(payload, args.output)
        print(f"Wrote diagnostics to {args.output}")
        return

    scene = load_gaussians_ply(ply_path)
    views = load_training_views(transforms_path, opengl_c2w=args.opengl_c2w)
    occlusion_maps = None
    if args.visibility == "gs-depth":
        from prf.gs_depth import render_training_depth_maps

        print(f"Rendering GS expected depth for {len(views)} training views on {args.device}", flush=True)
        occlusion_maps = render_training_depth_maps(
            ply_path,
            views,
            device=args.device,
            cache_dir=args.depth_cache,
        )

    if args.reuse_patches is not None:
        field = recompute_observation_from_archive(
            _load_archive(args.reuse_patches),
            scene,
            views,
            cfg,
            occlusion_maps=occlusion_maps,
            source_ply=str(ply_path),
            source_transforms=str(transforms_path),
        )
    else:
        field = compute_from_paths(
            ply_path,
            transforms_path,
            cfg,
            opengl_c2w=args.opengl_c2w,
            occlusion_maps=occlusion_maps,
            visibility_model="gs_expected_depth" if occlusion_maps is not None else "frustum",
        )
    export_field(field, args.output)
    archive = _load_archive(args.output / "physical_resolution_field.npz")
    payload = {"field": diagnose_field_archive(archive)}
    if not args.skip_rebuild_geometry:
        payload["rebuilt"] = diagnose_rebuilt_geometry(scene, cfg)
    write_diagnostics(payload, args.output)
    if args.spot_check:
        write_region_spot_checks(archive, payload["field"]["samples"], scene, views, transforms_path, args.output)
    if args.compare_to is not None:
        comparison = compare_observation_archives(_load_archive(args.compare_to), archive)
        (args.output / "occlusion_delta.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison, indent=2))
    if args.compare_stats_to is not None:
        stats = compare_field_statistics(_load_archive(args.compare_stats_to), archive)
        (args.output / "geometry_delta.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
    write_manifest(
        build_experiment_manifest(
            name=args.output.name,
            ply=ply_path,
            transforms=transforms_path,
            output=args.output,
            config=config_dict(cfg),
            visibility_model=field.visibility_model,
            reuse_patches=args.reuse_patches,
            depth_cache=args.depth_cache,
            extra={"summary": json.loads((args.output / "summary.json").read_text())},
        ),
        args.output,
    )
    finite = sum(1 for rec in field.records if math.isfinite(rec.R_phys_m_per_equiv_pixel))
    print(f"Wrote {len(field.records)} patches ({finite} finite) to {args.output}")


if __name__ == "__main__":
    main()
