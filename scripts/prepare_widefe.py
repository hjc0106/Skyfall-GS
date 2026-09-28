#!/usr/bin/env python3
"""Render one protocol-defined WideFE IDU camera-course episode.

Requires an activated Skyfall-GS Python environment, a compatible Gaussian
checkpoint, and the repository's native RaDe renderer. Writes the standard
FlowEdit prepared manifest, native RGB render images, and camera-course
provenance under --output-dir. FLUX is required by protocol for provenance but
is not loaded during preparation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

COURSE_KIND = "skyfall_flowedit_camera_course"
COURSE_SCHEMA_VERSION = 1
PREPARED_KIND = "flowedit_stage2_prepared"
RGB_CONTRACT = "normalized_float_clamped_v1"
RGB_SHA_SEMANTICS = "sha256(PIL RGB image.tobytes()) as written by render_flowedit_views"
APPEARANCE_ROW = 6
RENDER_PATH = "gaussian_renderer.render(testing=True)"
GRID_FORMULA = "np.linspace(-axis/2, +axis/2, grid_size + 2)[1:-1], meshgrid(x, y), z=0"
FLAT_FORMULA = "flat = view_index * samples_per_view + sample_index"
EPISODE_DIR_RE = re.compile(r"^episode_(\d+)_e(-?[0-9.]+)_r(-?[0-9.]+)$")


def _repo_root() -> Path:
    """Return the checkout containing this script, not a searched fallback."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "train.py").is_file():
        raise SystemExit(f"error: Skyfall-GS checkout not found at {root}")
    return root


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


# ---------------------------------------------------------------------------
# small IO helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | os.PathLike[str], what: str) -> dict:
    target = Path(path).expanduser()
    if not target.is_file():
        raise SystemExit(f"error: {what} not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: could not read {what} {target}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"error: {what} {target} must be a JSON object")
    return payload


def _write_json(path: str | os.PathLike[str], payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _require_value(protocol: dict, key: str, types, what: str):
    if key not in protocol:
        raise SystemExit(f"error: protocol is missing {key!r} ({what})")
    value = protocol[key]
    if isinstance(value, bool) or not isinstance(value, types):
        raise SystemExit(f"error: protocol {key!r} must be {what}, got {value!r}")
    return value


def _require_positive_float(protocol: dict, key: str, what: str) -> float:
    value = _require_value(protocol, key, (int, float), what)
    value = float(value)
    if not value > 0.0:
        raise SystemExit(f"error: protocol {key!r} must be > 0, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# protocol-derived course description
# ---------------------------------------------------------------------------


def _courses(protocol: dict) -> list[tuple[float, float]]:
    elevations = _require_value(protocol, "elevations", list, "curriculum elevations")
    radii = _require_value(protocol, "radii", list, "curriculum radii")
    if len(elevations) != len(radii) or not elevations:
        raise SystemExit(
            f"error: protocol elevations/radii must be equal-length non-empty lists, "
            f"got {len(elevations)} and {len(radii)}"
        )
    return [(float(e), float(r)) for e, r in zip(elevations, radii)]


def _grid_targets(grid_size: int, grid_width: float, grid_height: float) -> tuple[list[float], list[float], list[list[float]]]:
    """train.py's IDU look-at lattice, including its target ordering."""

    x = np.linspace(-grid_width / 2.0, grid_width / 2.0, grid_size + 2)[1:-1]
    y = np.linspace(-grid_height / 2.0, grid_height / 2.0, grid_size + 2)[1:-1]
    xx, yy = np.meshgrid(x, y)
    targets = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3).tolist()
    if len(targets) != grid_size * grid_size:
        raise SystemExit(f"error: look-at grid produced {len(targets)} targets for size {grid_size}")
    return [float(value) for value in x.tolist()], [float(value) for value in y.tolist()], [[float(v) for v in row] for row in targets]


def _resolve_course(protocol: dict, elevation: float, radius: float, episode_index: int) -> int:
    courses = _courses(protocol)
    matches = [index for index, (ele, rad) in enumerate(courses) if abs(ele - elevation) < 1e-9 and abs(rad - radius) < 1e-9]
    if not matches:
        raise SystemExit(
            f"error: (elevation={elevation:g}, radius={radius:g}) is not one of the protocol courses "
            f"{[f'e{e:g}/r{r:g}' for e, r in courses]}"
        )
    course_index = matches[0]
    if course_index != episode_index:
        raise SystemExit(
            f"error: --episode-index {episode_index} does not match the protocol course index "
            f"{course_index} of e{elevation:g}/r{radius:g}; angle episodes follow protocol order "
            f"{[f'e{e:g}/r{r:g}' for e, r in courses]}"
        )
    return course_index


def _check_episode_dirname(output_dir: Path, episode_index: int, elevation: float, radius: float) -> None:
    """Refuse an episode directory that names a different course."""

    match = EPISODE_DIR_RE.match(output_dir.name)
    if match is None:
        return
    index, ele, rad = int(match.group(1)), float(match.group(2)), float(match.group(3))
    if index != episode_index or abs(ele - elevation) > 1e-9 or abs(rad - radius) > 1e-9:
        raise SystemExit(
            f"error: --output-dir {output_dir} names episode {index} e{ele:g}/r{rad:g}, but the "
            f"requested episode is {episode_index} e{elevation:g}/r{radius:g}"
        )


def _grid_options(protocol: dict) -> dict:
    grid_size = _require_value(protocol, "grid_size", int, "IDU look-at grid size")
    grid_width = _require_positive_float(protocol, "grid_width", "IDU look-at grid width")
    grid_height = _require_positive_float(protocol, "grid_height", "IDU look-at grid height")
    cameras_per_target = _require_value(protocol, "cameras_per_target", int, "orbit poses per target")
    samples_per_pose = _require_value(protocol, "samples_per_pose", int, "independent reviews per pose")
    raster = _require_value(protocol, "raster", int, "native render raster")
    fov = _require_positive_float(protocol, "fov", "camera FoV in degrees")
    for name, value in (("grid_size", grid_size), ("cameras_per_target", cameras_per_target),
                        ("samples_per_pose", samples_per_pose), ("raster", raster)):
        if value < 1:
            raise SystemExit(f"error: protocol {name!r} must be >= 1, got {value}")
    if raster % 16:
        raise SystemExit(
            f"error: protocol raster {raster} is not a multiple of 16; FlowEdit would crop it"
        )
    if grid_size < 1:
        raise SystemExit(f"error: protocol grid_size must be >= 1, got {grid_size}")
    return {
        "grid_size": grid_size, "grid_width": grid_width, "grid_height": grid_height,
        "cameras_per_target": cameras_per_target, "samples_per_pose": samples_per_pose,
        "raster": raster, "fov_degrees": fov,
    }


def _sampler_block(protocol: dict) -> dict:
    sampler = protocol.get("sampler")
    if sampler is None:
        return {}
    if not isinstance(sampler, dict):
        raise SystemExit("error: protocol 'sampler' must be an object")
    return {str(key): value for key, value in sampler.items()}


def _flux_model_path(protocol: dict) -> str:
    value = protocol.get("flux_model_path") or ""
    if not isinstance(value, str) or not value.strip():
        raise SystemExit("error: protocol 'flux_model_path' must be a non-empty local directory")
    path = Path(value).expanduser()
    if not (path / "model_index.json").is_file():
        raise SystemExit(f"error: FLUX pipeline config missing: {path / 'model_index.json'}")
    return str(path.resolve())


def _resolve_checkpoint(protocol: dict, requested: str | None, episode_index: int) -> tuple[Path, str, str]:
    """Verify the base checkpoint or the immediately preceding completed episode."""
    if "base_checkpoint" not in protocol or "base_sha256" not in protocol:
        raise SystemExit("error: protocol requires base_checkpoint and base_sha256")
    base = Path(protocol["base_checkpoint"]).expanduser().resolve()
    resolved = Path(requested or base).expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"error: input checkpoint not found: {resolved}")
    if episode_index == 0:
        if resolved != base:
            raise SystemExit("error: the first course must start from the verified Stage 1 base")
        expected = protocol["base_sha256"]
        source = "verified_stage1"
    else:
        if episode_index < 0:
            raise SystemExit(f"error: episode index must be >= 0, got {episode_index}")
        if not isinstance(protocol.get("output_dir"), str) or not protocol["output_dir"].strip():
            raise SystemExit("error: protocol requires output_dir for predecessor checkpoint verification")
        state_path = Path(protocol["output_dir"]).expanduser() / "run_status.json"
        state = _load_json(state_path, "curriculum status")
        completed = state.get("episodes")
        if not isinstance(completed, list):
            raise SystemExit(f"error: {state_path} must contain an episodes list")
        predecessors = [item for item in completed
                        if isinstance(item, dict) and item.get("episode") == episode_index - 1]
        if len(predecessors) != 1:
            raise SystemExit("error: the immediately preceding IDU episode is not uniquely recorded")
        previous = predecessors[0]
        previous_checkpoint = Path(str(previous.get("output_checkpoint", ""))).expanduser().resolve()
        if resolved != previous_checkpoint:
            raise SystemExit("error: the next course must render the preceding IDU checkpoint")
        expected = previous.get("output_sha256")
        source = "previous_completed_idu"
    digest = _sha256_file(resolved)
    if not isinstance(expected, str) or digest != expected:
        raise SystemExit(f"error: input checkpoint hash changed: {resolved}")
    return resolved, source, digest


# ---------------------------------------------------------------------------
# camera course
# ---------------------------------------------------------------------------


def _build_course(options: dict, elevation: float, radius: float):
    """CameraInfos plus per-view pose provenance for one curriculum angle."""

    from utils.camera_utils import gen_idu_orbit_camera

    x_offsets, y_offsets, targets = _grid_targets(
        options["grid_size"], options["grid_width"], options["grid_height"]
    )
    infos, records = [], []
    for target_index, target in enumerate(targets):
        poses = gen_idu_orbit_camera(
            target,
            elevation=elevation,
            radius=radius,
            num_cams=options["cameras_per_target"],
            num_samples=1,
            height=options["raster"],
            width=options["raster"],
            fov=options["fov_degrees"],
        )
        if len(poses) != options["cameras_per_target"]:
            raise SystemExit(
                f"error: gen_idu_orbit_camera returned {len(poses)} poses, expected "
                f"{options['cameras_per_target']}"
            )
        for pose_index, info in enumerate(poses):
            index = len(infos)
            infos.append(info._replace(uid=1000 + index, image_name=f"idu_view_{index:05d}.png"))
            records.append({
                "index": index,
                "uid": 1000 + index,
                "target_index": target_index,
                "pose_index": pose_index,
                "target": list(target),
                "theta_degrees": 360.0 * pose_index / options["cameras_per_target"],
                "image_name": f"idu_view_{index:05d}.png",
            })
    return x_offsets, y_offsets, targets, infos, records


def _camera_list(infos, namespace):
    from arguments import ModelParams
    from utils.camera_utils import cameraList_from_camInfos

    dataset = ModelParams(argparse.ArgumentParser(add_help=False)).extract(namespace)
    return cameraList_from_camInfos(infos, 1, dataset, is_idu=True)


# ---------------------------------------------------------------------------
# render + refinement options
# ---------------------------------------------------------------------------


def _refine_options(*, samples_per_view: int, flux_model_path: str, device: str, sampler: dict):
    """The repo's own IDU option defaults, pinned to the Wide FlowEdit protocol."""

    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = argparse.ArgumentParser(add_help=False)
    ModelParams(parser)
    optim = OptimizationParams(parser)
    PipelineParams(parser)
    options = optim.extract(parser.parse_args([]))

    def _sampler_value(key: str, fallback):
        value = sampler.get(key, fallback)
        if value is None:
            return fallback
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"error: protocol sampler {key!r} must be a number, got {value!r}")
        return value

    options.idu_num_samples_per_view = int(samples_per_view)
    options.idu_model_type = "FLUX"
    options.flux_model_path = str(flux_model_path)
    options.idu_refine_device = str(device)
    options.idu_flow_edit_n_min = int(_sampler_value("n_min", options.idu_flow_edit_n_min))
    options.idu_flow_edit_n_max = int(_sampler_value("n_max", options.idu_flow_edit_n_max))
    n_max_end = sampler.get("n_max_end", None)
    options.idu_flow_edit_n_max_end = -1 if n_max_end is None else int(n_max_end)
    options.idu_flow_edit_n_avg = int(_sampler_value("n_avg", options.idu_flow_edit_n_avg))
    return options


def _render_views(context):
    from gaussian_renderer import render
    from refinement.types import RenderBundle

    def render_view(camera):
        package = render(
            camera, context["gaussians"], context["pipe"], context["background"],
            kernel_size=context["kernel_size"], testing=True,
        )
        return RenderBundle(
            rgb=package["render"], depth=package.get("render_depth"),
            alpha=package.get("render_alpha"), camera=camera,
        )

    return render_view


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render one WideFE IDU episode using the protocol camera course and native RaDe "
            "rasterizer; requires an activated Skyfall-GS environment and checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protocol", required=True, help="Canonical WideFE protocol JSON.")
    parser.add_argument("--checkpoint", default=None,
                        help="Stage 1 base checkpoint .pth; defaults to protocol['base_checkpoint'].")
    parser.add_argument("--output-dir", required=True, help="Episode directory (created if needed).")
    parser.add_argument("--episode-index", required=True, type=int,
                        help="Curriculum episode index; must equal the protocol course index.")
    parser.add_argument("--elevation", required=True, type=float, help="Curriculum elevation in degrees.")
    parser.add_argument("--radius", required=True, type=float, help="Curriculum orbit radius.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = "cuda:0"
    started = time.perf_counter()

    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = _load_json(protocol_path, "protocol")
    protocol_sha256 = _sha256_file(protocol_path)
    options = _grid_options(protocol)
    if args.episode_index < 0:
        raise SystemExit(f"error: --episode-index must be >= 0, got {args.episode_index}")
    _resolve_course(protocol, args.elevation, args.radius, args.episode_index)
    output_dir = Path(args.output_dir).expanduser().resolve()
    _check_episode_dirname(output_dir, args.episode_index, args.elevation, args.radius)
    checkpoint, checkpoint_source, checkpoint_sha256 = _resolve_checkpoint(protocol, args.checkpoint, args.episode_index)
    flux_model_path = _flux_model_path(protocol)
    source_path = protocol.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise SystemExit("error: protocol 'source_path' must be a non-empty string")
    sampler = _sampler_block(protocol)

    # Deterministic camera course: gen_idu_orbit_camera draws from the global
    # RNG even though its theta offset is pinned to zero today.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    previous = output_dir / "flowedit_prepared.json"
    if previous.is_file():
        try:
            existing = json.loads(previous.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("checkpoint") not in (None, str(checkpoint)):
            print(
                f"[prepare-flowedit] WARNING: {previous} was rendered from "
                f"{existing.get('checkpoint')!r}; overwriting with {str(checkpoint)!r}",
                flush=True,
            )

    from refinement.scene_zoom import SceneZoomConfig, build_arg_namespace, load_scene_context
    from refinement.flowedit_stage2 import render_flowedit_views

    cfg = SceneZoomConfig(start_checkpoint=str(checkpoint), output_dir=str(output_dir), source_path=source_path)
    namespace = build_arg_namespace(cfg)
    print(f"[prepare-flowedit] loading checkpoint scene {checkpoint}", flush=True)
    context = load_scene_context(cfg, namespace)
    backend = getattr(context["pipe"], "rasterizer_backend", None)
    if backend != "rade":
        raise SystemExit(
            f"error: the checkpoint dataset contract selects rasterizer_backend={backend!r}; this run "
            "requires the native RaDe rasterizer ('rade') from the Stage 1 cfg"
        )
    appearance_embeddings = getattr(context["gaussians"], "appearance_embeddings", None)
    appearance_rows = 0 if appearance_embeddings is None else len(appearance_embeddings)
    appearance_row_used = min(APPEARANCE_ROW, appearance_rows - 1) if appearance_rows else 0
    if appearance_rows and appearance_row_used != APPEARANCE_ROW:
        print(
            f"[prepare-flowedit] WARNING: the checkpoint carries {appearance_rows} appearance rows; the "
            f"native testing render uses row {appearance_row_used}, not {APPEARANCE_ROW}",
            flush=True,
        )
    render_provenance = {
        "path": RENDER_PATH,
        "rasterizer_backend": backend,
        "kernel_size": float(context["kernel_size"]),
        "first_iter": int(context["first_iter"]),
        "filter_3d_source": str(context["filter_3d_source"]),
        "appearance_row": APPEARANCE_ROW,
        "appearance_row_used": appearance_row_used,
        "appearance_rows": appearance_rows,
        "rgb_contract": RGB_CONTRACT,
        "rgb_sha256_semantics": RGB_SHA_SEMANTICS,
        "deterministic_camera_uid_base": 1000,
    }
    print(
        f"[prepare-flowedit] rasterizer_backend=rade kernel_size={context['kernel_size']} "
        f"filter_3d={context['filter_3d_source']}",
        flush=True,
    )

    x_offsets, y_offsets, targets, infos, pose_records = _build_course(
        options, args.elevation, args.radius
    )
    views = _camera_list(infos, namespace)
    if len(views) != len(pose_records):
        raise SystemExit(f"error: built {len(views)} cameras for {len(pose_records)} poses")
    print(
        f"[prepare-flowedit] episode {args.episode_index:02d} e{args.elevation:g}/r{args.radius:g}: "
        f"{len(targets)} look-at targets x {options['cameras_per_target']} poses = {len(views)} views",
        flush=True,
    )

    refine_options = _refine_options(
        samples_per_view=options["samples_per_pose"], flux_model_path=flux_model_path,
        device=device, sampler=sampler,
    )

    try:
        prepared = render_flowedit_views(
            views, _render_views(context),
            checkpoint_path=str(checkpoint), episode_dir=str(output_dir),
            episode_idx=int(args.episode_index), options=refine_options,
        )
    finally:
        del context, views, infos
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if prepared.get("kind") != PREPARED_KIND or prepared.get("width") != options["raster"]:
        raise SystemExit(
            f"error: render_flowedit_views produced kind={prepared.get('kind')!r} raster="
            f"{prepared.get('width')}x{prepared.get('height')}; expected {PREPARED_KIND} at {options['raster']}"
        )

    # Merge the pose provenance with the manifest's verified per-view records.
    manifest_views = prepared.get("views") or []
    if len(manifest_views) != len(pose_records):
        raise SystemExit(
            f"error: manifest recorded {len(manifest_views)} views for {len(pose_records)} rendered poses"
        )
    course_views = []
    for record, view in zip(pose_records, manifest_views):
        if int(view["index"]) != record["index"] or int(view["uid"]) != record["uid"]:
            raise SystemExit(
                f"error: manifest view order changed (index {view['index']}/uid {view['uid']} vs "
                f"{record['index']}/{record['uid']})"
            )
        course_views.append({
            **record,
            "render_path": str(view["render_path"]),
            "rgb_sha256": str(view["rgb_sha256"]),
        })

    course = {
        "schema_version": COURSE_SCHEMA_VERSION,
        "kind": COURSE_KIND,
        "created_at": _utc_now(),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "prepared_manifest": str(output_dir / "flowedit_prepared.json"),
        "prepared_kind": prepared["kind"],
        "prepared_schema_version": prepared["schema_version"],
        "checkpoint": str(checkpoint),
        "checkpoint_source": checkpoint_source,
        "checkpoint_stat": prepared["checkpoint_stat"],
        "checkpoint_sha256": checkpoint_sha256,
        "episode_idx": int(args.episode_index),
        "episode_dir": str(output_dir),
        "elevation": float(args.elevation),
        "radius": float(args.radius),
        "fov_degrees": options["fov_degrees"],
        "raster": options["raster"],
        "camera_motion": prepared["camera_motion"],
        "grid": {
            "size": options["grid_size"],
            "width": options["grid_width"],
            "height": options["grid_height"],
            "x_offsets": x_offsets,
            "y_offsets": y_offsets,
            "formula": GRID_FORMULA,
        },
        "targets": targets,
        "cameras_per_target": options["cameras_per_target"],
        "samples_per_pose": options["samples_per_pose"],
        "num_views": prepared["num_views"],
        "num_images": prepared["num_views"] * options["samples_per_pose"],
        "flat_index_formula": FLAT_FORMULA,
        "render": render_provenance,
        "sampler": sampler,
        "model_type": "FLUX",
        "flux_model_path": flux_model_path,
        "device": device,
        "views": course_views,
    }
    course_path = output_dir / "flowedit_camera_course.json"
    _write_json(course_path, course)

    summary = {
        "status": "complete",
        "episode_idx": course["episode_idx"],
        "episode_dir": course["episode_dir"],
        "elevation": course["elevation"],
        "radius": course["radius"],
        "checkpoint": course["checkpoint"],
        "prepared_manifest": course["prepared_manifest"],
        "course_manifest": str(course_path),
        "num_views": course["num_views"],
        "num_images": course["num_images"],
        "raster": course["raster"],
        "seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(summary, indent=2), flush=True)
    print("FLOWEDIT_PREPARE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
