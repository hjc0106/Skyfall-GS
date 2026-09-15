#!/usr/bin/env python3
"""Offline local scene showcase renderer (owner: LocalSceneBackend).

Renders fresh CUDA orbit videos for every complete archived Skyfall-GS stage
model in ``/dataset/Skyfall-GS/experiments/stage2_gaussianzoom_20260913`` on the
shared checked-in ``r328_e17_fov20`` camera path (native 1024x1024, all original
frames, original fps), copies the locally available original released Stage2 and
GES reference videos into the output, and writes a ``catalog.json`` +
``provenance.json`` describing every asset.  Parent Main consumes the catalog to
assemble the final showcase video; no HTTP server or frontend lives here.

Faithfulness rules (mirrors scripts/evaluate_dataset.py):
  * models are loaded with the exact ``load_archive_model`` archive loader
    (checkpoint + filtered PLY, CPU checkpoint tensors, GPU rendering state,
    optimizer state omitted);
  * learned appearance is preserved: the fixed training appearance row
    ``min(6, n_train - 1)`` -- the formal evaluator's testing branch default --
    is passed explicitly to the renderer;
  * rendering reuses ``gaussian_renderer.render`` (filter3D + appearance MLP);
  * camera math reuses ``render_video.get_path_from_json`` conventions
    (OpenGL -> COLMAP flip) but builds ``MiniCam`` directly so no blank GT
    raster is allocated for path cameras;
  * supplementary heldout metrics are computed fresh with ``MetricStack`` into
    this output root only; formal archive evaluation results are never touched.

Usage:
    python scripts/local_scene_comparison.py build --output-root ROOT
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from scripts import evaluate_dataset as ev

CATALOG_SCHEMA = "local_scene_comparison_catalog_v1"
DEFAULT_MANIFEST = str(ev.DEFAULT_MANIFEST)
STAGE_EXPECTED_ITERATION = {"stage1": 30000, "stage2": 80000}
PREFERRED_TRAJECTORY = "r328_e17_fov20"
SAMPLE_FRAMES = 4
LPIPS_NET = "alex"


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    return ev.sha256_file(path)


def free_cuda() -> None:
    ev.free_cuda()


def log(msg: str) -> None:
    print(f"[local_scene_comparison {now_iso()}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# roster discovery from the verified archive
# --------------------------------------------------------------------------- #
def discover_roster(manifest: dict[str, Any], archive_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Complete stages only: the stage's own on-disk archive marker must be
    verified and its final iteration must equal the expected value (stage1
    30000 / stage2 80000).  Formal evaluation is NOT an eligibility gate: the
    per-scene rollup can be stale, so catalog metrics come from this build's
    fresh MetricStack computation; any archived heldout aggregate is recorded
    in provenance for reference only."""
    summary = {entry["scene"]: entry for entry in manifest["scenes"]}
    roster: list[dict[str, Any]] = []
    notes: list[str] = []
    for scene_id, scene_entry in summary.items():
        for stage_id in ("stage1", "stage2"):
            stage_dir = archive_root / scene_id / stage_id
            state = ev.archive_state(stage_dir)
            if not state.get("ready"):
                if scene_id.startswith("JAX"):
                    notes.append(
                        f"{scene_id}/{stage_id}: not eligible (on-disk archive marker "
                        f"{'missing' if not state.get('present') else 'not verified'}); excluded."
                    )
                continue
            _ckpt, _ply, iteration = ev.find_final_assets(stage_dir)
            expected = STAGE_EXPECTED_ITERATION[stage_id]
            if iteration != expected:
                notes.append(
                    f"{scene_id}/{stage_id}: final iteration {iteration} != expected {expected}; excluded."
                )
                continue
            eval_status = ev.evaluation_state(archive_root / scene_id / "evaluation" / stage_id)
            eval_payload = eval_status.get("payload") or {}
            roster.append(
                {
                    "scene": scene_id,
                    "stage": stage_id,
                    "stage_dir": stage_dir,
                    "iteration": iteration,
                    "checkpoint_sha256": ev.archived_asset_ids(stage_dir, stage_id).get("checkpoint", {}).get("sha256"),
                    "ply_sha256": ev.archived_asset_ids(stage_dir, stage_id).get("point_cloud", {}).get("sha256"),
                    "num_gaussians": eval_payload.get("num_gaussians"),
                    "formal_heldout_aggregate": eval_payload.get("heldout_aggregate"),
                    "dataset": scene_entry.get("dataset"),
                }
            )
    return roster, notes


def camera_path_for_scene(manifest: dict[str, Any], scene: str) -> dict[str, Any]:
    """Shared checked-in trajectory; ``r328_e17_fov20`` when the scene has it,
    otherwise the scene's lowest-radius available orbit (recorded verbatim)."""
    candidates = ev.camera_path_files(manifest, scene)
    if not candidates:
        raise FileNotFoundError(f"no checked-in camera paths for {scene}")
    chosen = next((p for p in candidates if p.stem == PREFERRED_TRAJECTORY), candidates[0])
    fallback = chosen.stem != PREFERRED_TRAJECTORY
    return {
        "path_json": chosen,
        "trajectory": chosen.stem,
        "is_fallback": fallback,
        "candidates": [p.stem for p in candidates],
    }


# --------------------------------------------------------------------------- #
# cameras: checked-in path -> MiniCam, no blank GT raster allocation
# --------------------------------------------------------------------------- #
def _three_js_focal(fov_deg: float, image_height: int) -> float:
    pp_h = image_height / 2.0
    return pp_h / math.tan(math.radians(fov_deg) / 2.0)


def minicams_from_path_json(payload: dict[str, Any], device: torch.device) -> list[Any]:
    """Mirror render_video.get_path_from_json pose math without Camera objects.

    Every frame becomes a ``MiniCam`` (world_view/full_proj already transposed,
    on CUDA) so the renderer never allocates a full blank GT raster.
    """
    from scene.cameras import MiniCam
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    height = int(payload["render_height"])
    width = int(payload["render_width"])
    cams: list[MiniCam] = []
    for idx, entry in enumerate(payload["camera_path"]):
        c2w = np.asarray(entry["camera_to_world"], dtype=np.float64).reshape(4, 4)
        # OpenGL/Blender camera axes (Y up, Z back) -> COLMAP (Y down, Z forward).
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        focal = _three_js_focal(float(entry["fov"]), height)
        fov_y = focal2fov(focal, height)
        fov_x = focal2fov(focal, width)
        znear, zfar = 0.01, 100.0
        world_view = torch.tensor(getWorld2View2(R, T), dtype=torch.float32, device=device).transpose(0, 1)
        projection = getProjectionMatrix(
            znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y, cx=0.0, cy=0.0
        ).to(device).transpose(0, 1)
        full_proj = (world_view.unsqueeze(0).bmm(projection.unsqueeze(0))).squeeze(0)
        cams.append(
            MiniCam(
                width=width,
                height=height,
                fovy=fov_y,
                fovx=fov_x,
                znear=znear,
                zfar=zfar,
                world_view_transform=world_view,
                full_proj_transform=full_proj,
            )
        )
    return cams


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def appearance_row(gaussians: Any) -> tuple[Any, int | None]:
    """Fixed training appearance row min(6, n_train-1): the formal evaluator's
    testing-branch default, passed explicitly so it cannot fall back to the
    mean embedding."""
    if not gaussians.appearance_enabled:
        return None, None
    embeddings = gaussians.appearance_embeddings
    row = min(6, len(embeddings) - 1)
    return embeddings[row].detach(), row


def rel_path(path: Path, root: Path) -> str:
    """Output-root-relative asset path; catalog URLs are relative by contract."""
    return str(Path(path).relative_to(Path(root)))


@torch.no_grad()
def render_orbit_video(
    model: ev.LoadedModel,
    cams: list[Any],
    appearance_embedding: Any,
    video_path: Path,
    frames_dir: Path,
    fps: float,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    import mediapy as media
    from PIL import Image

    from gaussian_renderer import render

    frames_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(cams)
    sample_indices = [i * num_frames // SAMPLE_FRAMES for i in range(SAMPLE_FRAMES)]
    background = torch.tensor(
        [1.0, 1.0, 1.0] if model.dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    gaussians = model.gaussians
    started = time.perf_counter()
    first_shape: tuple[int, ...] | None = None
    sample_paths: list[str] = []
    with media.VideoWriter(
        path=str(video_path), shape=(int(cams[0].image_height), int(cams[0].image_width)), fps=float(fps)
    ) as writer:
        for idx, cam in enumerate(cams):
            pkg = render(
                cam,
                gaussians,
                model.pipe,
                background,
                model.dataset.kernel_size,
                testing=True,
                appearance_embedding=appearance_embedding,
            )
            image = torch.clamp(pkg["render"], 0.0, 1.0)
            arr = ev.tensor_to_uint8_hwc(image)
            if first_shape is None:
                first_shape = arr.shape
            writer.add_image(arr)
            if idx in sample_indices:
                out = frames_dir / f"frame_{idx:05d}.png"
                Image.fromarray(arr).save(out)
                sample_paths.append(rel_path(out, output_root))
            del pkg, image
    free_cuda()
    return {
        "video": rel_path(video_path, output_root),
        "num_frames": num_frames,
        "width": int(first_shape[1]) if first_shape else None,
        "height": int(first_shape[0]) if first_shape else None,
        "fps": float(fps),
        "sample_indices": sample_indices,
        "sample_frames": sample_paths,
        "render_seconds": time.perf_counter() - started,
    }


def poster_from_sample(frames_dir: Path, sample_paths: list[str], output_root: Path) -> str | None:
    """Poster = the ~25% sample frame re-encoded as JPEG (no extra GPU pass)."""
    if not sample_paths:
        return None
    from PIL import Image

    src = output_root / sample_paths[min(1, len(sample_paths) - 1)]
    poster = frames_dir / "poster.jpg"
    Image.open(src).convert("RGB").save(poster, quality=92)
    return rel_path(poster, output_root)


def heldout_metrics(
    model: ev.LoadedModel, metrics: ev.MetricStack, device: torch.device
) -> dict[str, Any] | None:
    """Supplementary heldout PSNR/SSIM/LPIPS/L1 computed fresh for this output
    root; never rewrites formal archive evaluation results."""
    try:
        per_view: list[dict[str, Any]] = []
        for camera in model.scene.getTestCameras():
            pkg = ev.render_camera(model, camera, device, testing=True)
            image = torch.clamp(pkg["render"], 0.0, 1.0)
            gt = torch.clamp(camera.original_image.to(device), 0.0, 1.0)
            values = metrics.reference(image, gt)
            per_view.append({"image_name": str(camera.image_name), **values})
            del pkg, image, gt
            free_cuda()
        aggregate: dict[str, Any] = {"num_views": len(per_view)}
        for key in ("psnr", "ssim", "lpips", "l1"):
            vals = [row[key] for row in per_view if row.get(key) is not None]
            aggregate[key] = float(np.mean(vals)) if vals else None
        return aggregate
    except Exception as exc:  # supplementary only: reported, not fatal
        log(f"heldout metrics failed: {type(exc).__name__}: {exc}")
        return None


def video_probe(path: Path) -> dict[str, Any]:
    """Frame count / fps / size of a video, via cv2 (cheap, no decode-all)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
# reference assets (original released videos, copied verbatim)
# --------------------------------------------------------------------------- #


def copy_reference_assets(manifest: dict[str, Any], scene: str, media_root: Path) -> tuple[dict[str, Any], list[str]]:
    eval_root = Path(manifest["local_eval_archive_root"]) if manifest.get("local_eval_archive_root") else None
    if eval_root is None:
        eval_root = Path("/dataset/Skyfall-GS/eval_data")
    scene_eval = eval_root / "data_eval_JAX" / scene
    out_dir = media_root / "reference" / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    reference: dict[str, Any] = {"trajectory": PREFERRED_TRAJECTORY}
    notes: list[str] = []
    src = scene_eval / "ours_stage2" / f"{PREFERRED_TRAJECTORY}.mp4"
    if src.is_file():
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        reference["video"] = rel_path(dst, media_root.parent)
        reference["video_source"] = str(src)
        reference["video_sha256"] = sha256_file(dst)
        reference["video_probe"] = video_probe(dst)
    else:
        reference["video"] = None
        notes.append(f"{scene}: no local original released Stage2 video at {src}.")
    reference["alignment_note"] = (
        "Local original released Stage2 video copied verbatim for labelled comparison; the "
        "GT/*.mp4 asset is the GES reference, NOT an aligned real-photo ground truth. "
        "Sharing the public trajectory and synchronized progress does not prove GT camera "
        "correspondence; lighting/appearance differences also affect pixel appearance."
    )
    gt_src = scene_eval / "GT" / f"{scene}_lower_merged.mp4"
    if gt_src.is_file():
        dst = out_dir / gt_src.name
        shutil.copy2(gt_src, dst)
        reference["gt_video"] = rel_path(dst, media_root.parent)
        reference["gt_video_source"] = str(gt_src)
        reference["gt_video_sha256"] = sha256_file(dst)
        reference["gt_video_probe"] = video_probe(dst)
    else:
        reference["gt_video"] = None
        notes.append(f"{scene}: no local GES reference video at {gt_src}.")
    return reference, notes


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    fh = open(output_root / ".local_scene_comparison_build.lock", "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        fh.close()
        raise RuntimeError(f"another build already holds the lock in {output_root}: {exc}") from exc
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} started={now_iso()}\n")
    fh.flush()
    return fh


def write_build_status(output_root: Path, payload: dict[str, Any]) -> None:
    ev.write_json(output_root / "build_status.json", payload)


def cmd_build(args: argparse.Namespace) -> int:
    manifest = ev.load_manifest(Path(args.manifest))
    archive_root = Path(args.archive_root or manifest["archive_root"])
    output_root = Path(args.output_root).resolve()
    lock = build_lock(output_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA unavailable: this renderer is a faithful CUDA pipeline, refusing CPU fallback")
    media_root = output_root / "media"
    roster, roster_notes = discover_roster(manifest, archive_root)
    if not roster:
        raise RuntimeError("no complete verified stages discovered; refusing to emit an empty catalog")

    notes: list[str] = list(roster_notes)
    notes.extend(
        [
            "All orbit clips are fresh CUDA inference from the local archived models; "
            "no cached or prerecorded fallback is used.",
            "Learned appearance is preserved with the fixed training appearance row "
            "min(6, n_train-1), identical to the formal evaluator's testing branch.",
            "Supplementary heldout metrics are computed fresh into this output root only; "
            "formal archive evaluation results are untouched.",
            "Original released videos copied under media/reference are labelled comparisons "
            "from the official Skyfall-GS release; the GT/*.mp4 files are the GES reference, "
            "NOT an aligned real-photo ground truth.",
            "Sharing a public camera trajectory and synchronized progress does not prove GT "
            "camera correspondence; lighting/appearance differences also affect pixels.",
            "An original Stage1 baseline of OUR models is not the original Skyfall-GS Stage1.",
            "JAX_264 has no stage2 in the archive; it ships stage1 only.",
        ]
    )

    metrics = ev.MetricStack(device, lpips_net=LPIPS_NET)
    # Group roster by scene, stages ordered stage1 -> stage2.
    scenes: dict[str, list[dict[str, Any]]] = {}
    for entry in roster:
        scenes.setdefault(entry["scene"], []).append(entry)
    for entry_list in scenes.values():
        entry_list.sort(key=lambda e: e["stage"])

    scene_payloads: list[dict[str, Any]] = []
    provenance_scenes: list[dict[str, Any]] = []
    completed = 0
    total = sum(len(v) for v in scenes.values())
    started_at = now_iso()
    write_build_status(
        output_root,
        {"status": "running", "started_at": started_at, "stages_total": total,
         "stages_completed": 0, "scenes": sorted(scenes)},
    )

    for scene_id, stage_entries in sorted(scenes.items()):
        path_info = camera_path_for_scene(manifest, scene_id)
        path_json = path_info["path_json"]
        payload = ev.read_json(path_json)
        path_sha = sha256_file(path_json)
        fps = float(payload.get("fps", 24))
        reference, ref_notes = copy_reference_assets(manifest, scene_id, media_root)
        has_reference = reference.get("video") is not None
        status_notes = list(ref_notes)
        if path_info["is_fallback"]:
            status_notes.append(
                f"no {PREFERRED_TRAJECTORY} path for this scene; rendered on {path_info['trajectory']}."
            )

        stage_payloads: list[dict[str, Any]] = []
        prov_stages: list[dict[str, Any]] = []
        for entry in stage_entries:
            stage_id = entry["stage"]
            stage_dir = Path(entry["stage_dir"])
            log(f"loading {scene_id}/{stage_id} (iteration {entry['iteration']})")
            try:
                model, load_report = ev.load_archive_model(manifest, scene_id, stage_dir, device, do_hash=False)
            except Exception as exc:
                raise RuntimeError(f"failed to load {scene_id}/{stage_id} from {stage_dir}: {exc}") from exc
            cams = minicams_from_path_json(payload, device)
            if len(cams) != len(payload["camera_path"]):
                raise RuntimeError(f"camera build mismatch for {scene_id}")
            out_dir = media_root / "orbit" / scene_id / stage_id
            out_dir.mkdir(parents=True, exist_ok=True)
            video_path = out_dir / f"orbit_{scene_id}_{stage_id}_{path_info['trajectory']}.mp4"
            try:
                emb, emb_row = appearance_row(model.gaussians)
                meta = render_orbit_video(
                    model, cams, emb, video_path, out_dir, fps, device, output_root
                )
                poster = poster_from_sample(out_dir, meta["sample_frames"], output_root)
                heldout = heldout_metrics(model, metrics, device)
            finally:
                del cams
                del model
                free_cuda()
            stage_video = rel_path(video_path, output_root) if video_path.is_file() else None
            if video_path.is_file():
                video_probe_data = video_probe(video_path)
                if video_probe_data["frames"] != meta["num_frames"]:
                    raise RuntimeError(
                        f"{scene_id}/{stage_id}: encoded video has {video_probe_data['frames']} frames "
                        f"vs {meta['num_frames']} rendered frames."
                    )
            else:
                raise RuntimeError(f"orbit video was not produced for {scene_id}/{stage_id}")
            stage_payloads.append(
                {
                    "id": stage_id,
                    "iteration": entry["iteration"],
                    "gaussians": entry.get("num_gaussians")
                    or load_report["checkpoint_contents"]["num_gaussians"],
                    "model_identity": {
                        "checkpoint_sha256": entry["checkpoint_sha256"],
                        "ply_sha256": entry["ply_sha256"],
                    },
                    "metrics": heldout,
                    "trajectory": path_info["trajectory"],
                    "video": stage_video,
                    "poster": poster,
                    "sample_frames": meta["sample_frames"],
                }
            )
            prov_stages.append(
                {
                    "stage": stage_id,
                    "iteration": entry["iteration"],
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "ply_sha256": entry["ply_sha256"],
                    "num_gaussians": entry.get("num_gaussians")
                    or load_report["checkpoint_contents"]["num_gaussians"],
                    "formal_heldout_aggregate": entry.get("formal_heldout_aggregate"),
                    "appearance_policy": (
                        "fixed training appearance row "
                        f"min(6, n_train-1) = row {emb_row}"
                        if emb_row is not None
                        else "model has no learned appearance embedding"
                    ),
                    "camera_path": {
                        "trajectory": path_info["trajectory"],
                        "json_sha256": path_sha,
                        "num_frames": meta["num_frames"],
                        "radius": float(payload.get("_radius", 0.0)),
                        "elevation_deg": float(payload.get("_elevation", 0.0)),
                        "target": payload.get("_target"),
                        "render_size": [meta["width"], meta["height"]],
                        "fps": meta["fps"],
                    },
                    "orbit_video": stage_video,
                    "video_probe": video_probe_data,
                    "sample_indices": meta["sample_indices"],
                    "sample_frames": meta["sample_frames"],
                    "poster": poster,
                    "heldout_metrics": heldout,
                    "load_seconds": load_report.get("load_seconds"),
                    "render_seconds": meta["render_seconds"],
                }
            )
            completed += 1
            write_build_status(
                output_root,
                {
                    "status": "running",
                    "started_at": started_at,
                    "stages_total": total,
                    "stages_completed": completed,
                    "current": f"{scene_id}/{stage_id}",
                },
            )

        default_stage = "stage2" if any(s["id"] == "stage2" for s in stage_payloads) else (
            stage_payloads[0]["id"] if stage_payloads else None
        )
        scene_payloads.append(
            {
                "id": scene_id,
                "name": scene_id,
                "default_stage": default_stage,
                "stages": stage_payloads,
                "reference": reference,
                "comparison": {"video": None, "sheet": None},
                "status_notes": status_notes
                + ([] if has_reference else ["no local original released video for this scene; "
                                              "rendered outputs only."]),
            }
        )
        provenance_scenes.append({"scene": scene_id, "camera_path": path_info["trajectory"],
                                  "camera_path_json_sha256": path_sha, "stages": prov_stages})
        write_build_status(
            output_root,
            {"status": "running", "started_at": started_at, "stages_total": total,
             "stages_completed": completed, "last_scene": scene_id},
        )

    device_info = {
        "type": device.type,
        "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    catalog = {
        "schema_version": 1,
        "kind": "local_scene_comparison_catalog",
        "schema": CATALOG_SCHEMA,
        "generated_at": now_iso(),
        "device": device_info,
        "paths_are_relative_to_output_root": True,
        "appearance_policy": "fixed training appearance row min(6, n_train-1), as in the formal evaluator",
        "trajectory": PREFERRED_TRAJECTORY,
        "notes": notes,
        "scenes": scene_payloads,
    }
    ev.write_json(output_root / "catalog.json", catalog)
    provenance = {
        "schema": "local_scene_comparison_provenance_v1",
        "generated_at": now_iso(),
        "manifest": manifest.get("_manifest_path"),
        "archive_root": str(archive_root),
        "camera_paths_root": str(Path(manifest["local_project"]) / "camera_paths"),
        "device": device_info,
        "notes": notes,
        "scenes": provenance_scenes,
    }
    ev.write_json(output_root / "provenance.json", provenance)
    write_build_status(output_root, {"status": "completed", "started_at": started_at,
                                     "stages_total": total, "stages_completed": completed,
                                     "completed_at": now_iso()})
    try:
        (output_root / ".local_scene_comparison_build.lock").unlink(missing_ok=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    log(f"build complete: {completed}/{total} stages rendered; catalog at {output_root / 'catalog.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="render all complete archived stage models into an output root")
    build.add_argument("--output-root", required=True)
    build.add_argument("--manifest", default=DEFAULT_MANIFEST)
    build.add_argument("--archive-root", default=None, help="defaults to manifest archive_root")
    build.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
