"""Compact retention for Skyfall-GS Stage 1 / Stage 2 (GaussianZoom) runs.

A full Stage 2 episode writes several gigabytes of scratch that is consumed
inside the episode and never read again:

* ``geometry/``           per-view 3DGS rgb/depth/alpha renders of the novel views
* ``dloral/``             DLoRAL inputs/outputs for the refinement worker
* ``context/``            context view rasters handed to the VLM / refiner
* ``generation_cache/``   CachedRefiner cache (regenerable from the seeds/prompts)
* ``render_depth/``       MoGe depth targets (float32 .npy, consumed by the episode)

``render/``, ``render_refine/`` and ``render_after_train/`` hold one PNG per
generated sample; only a handful of representative views are useful once the
episode has finished.

This module keeps, per completed episode, a bounded set of small provenance and
visual artifacts (metrics, prompt/camera provenance, representative comparison
panels) and then removes the consumed scratch.  It never touches the final
checkpoint or its matching point cloud: those are the archive/evaluation
artifacts and are retired only by the training queue after a locally verified
archive exists (see ``scripts/train_full_dataset.py reap``).

Everything here is deterministic and safe to re-run: an episode that is already
compact is left untouched, and a missing input directory is reported instead of
raising.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from PIL import Image, ImageDraw

#: Number of representative sample indices retained per episode.
PANEL_VIEW_COUNT = 4
#: Tile edge used when composing comparison panels (matches the prior
#: single-scene summarizer, so panels stay comparable across runs).
PANEL_TILE = 512
PANEL_LABEL_HEIGHT = 40

#: Dense per-episode scratch consumed by the episode itself.
DENSE_EPISODE_DIRS = ("geometry", "dloral", "context", "generation_cache", "render_depth")
#: Per-sample image dirs (bulk pruned, representatives copied out first).
EPISODE_IMAGE_DIRS = ("render", "render_refine", "render_after_train")
#: Small per-episode provenance/files that are always retained.
KEEP_EPISODE_FILES = ("episode_meta.json", "stage2_prepared.json", "stage2_synthesis.json")
KEEP_EPISODE_DIRS = ("prompt_cache",)

COMPACT_SCHEMA_VERSION = 1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: os.PathLike | str, payload: dict) -> str:
    """Write ``payload`` as JSON to ``path`` via a temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return str(path)


def _tree_size(path: os.PathLike | str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def select_panel_indices(available: Sequence[int], count: int = PANEL_VIEW_COUNT) -> List[int]:
    """Pick evenly spaced representative indices from ``available``."""
    ordered = sorted(int(index) for index in available)
    if not ordered:
        return []
    if count <= 1 or len(ordered) <= count:
        return ordered
    span = len(ordered) - 1
    return sorted({ordered[round(step * span / (count - 1))] for step in range(count)})


def _sample_indices(episode_root: Path) -> List[int]:
    for directory in EPISODE_IMAGE_DIRS:
        path = episode_root / directory
        if not path.is_dir():
            continue
        indices = []
        for name in os.listdir(path):
            stem, _, suffix = name.partition(".")
            if suffix == "png" and stem.isdigit():
                indices.append(int(stem))
        if indices:
            return sorted(indices)
    return []


def _open_tile(path: Path, tile: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)


def build_comparison_panel(episode_root: Path, indices: Sequence[int], out_path: Path,
                           *, tile: int = PANEL_TILE, title: Optional[str] = None) -> dict:
    """Compose ``render | render_refine | render_after_train`` rows into one JPEG.

    Mirrors the layout of the prior single-scene summarizer so panels remain
    visually comparable.  Missing images are skipped rather than fabricated.
    """
    labels = [("render", "input to episode"), ("render_refine", "DLoRAL supervision"),
              ("render_after_train", "after 3DGS episode")]
    rows: List[Image.Image] = []
    missing: List[str] = []
    for index in indices:
        row = Image.new("RGB", (tile * len(labels), tile + PANEL_LABEL_HEIGHT), "white")
        draw = ImageDraw.Draw(row)
        for column, (folder, label) in enumerate(labels):
            source = episode_root / folder / f"{index:05d}.png"
            header = f"{episode_root.name} | {index:05d} | {label}"
            draw.text((column * tile + 8, 8), header, fill="black")
            if source.is_file():
                row.paste(_open_tile(source, tile), (column * tile, PANEL_LABEL_HEIGHT))
            else:
                missing.append(f"{folder}/{index:05d}.png")
        rows.append(row)
    if not rows:
        return {"path": None, "indices": [], "missing": missing, "bytes": 0}
    if title:
        banner = Image.new("RGB", (rows[0].width, 24), "white")
        ImageDraw.Draw(banner).text((8, 6), title, fill="black")
        rows.insert(0, banner)
    total_height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (rows[0].width, total_height), "white")
    offset = 0
    for row in rows:
        canvas.paste(row, (0, offset))
        offset += row.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=92)
    return {
        "path": str(out_path),
        "indices": list(indices),
        "missing": missing,
        "bytes": os.path.getsize(out_path),
    }


def _compact_compare_html(episode_root: Path, indices: Sequence[int], dropped: Sequence[str],
                          elevation, radius, episode_idx: int) -> str:
    rows = []
    for index in indices:
        cells = "".join(
            f'<td><img src="retained_views/{index:05d}_{folder}.png" /></td>'
            if (episode_root / "retained_views" / f"{index:05d}_{folder}.png").is_file()
            else "<td>-</td>"
            for folder in EPISODE_IMAGE_DIRS
        )
        rows.append(f"<tr><td>{index:05d}</td>{cells}</tr>")
    pruned = ", ".join(dropped) if dropped else "none"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<title>IDU episode {episode_idx:02d} (compact)</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 16px; font-family: ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; }}
  img {{ width: 280px; height: auto; background: #000; }}
  table {{ border-collapse: collapse; }}
  td, th {{ padding: 6px; vertical-align: top; }}
  code {{ color: #9cf; }}
</style></head><body>
<h2>Episode {episode_idx:02d} · e={elevation} · r={radius} (compact retention)</h2>
<p>columns: 3DGS render · GaussianZoom / DLoRAL supervision · after this episode's 3DGS training.
Only {len(indices)} representative sample(s) are retained; the per-sample dirs
<code>{pruned}</code> were removed after the episode completed (regenerable by re-running the
episode with the recorded seeds/prompts).</p>
<table>
<tr><th>id</th><th>render</th><th>render_refine</th><th>render_after_train</th></tr>
{''.join(rows)}
</table></body></html>
"""
    path = episode_root / "compare.html"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return str(path)


def _remove_path(path: Path) -> int:
    if not path.exists():
        return 0
    size = _tree_size(path) if path.is_dir() else path.stat().st_size
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return size


def _finish_episode_cleanup(episode_root: Path, summary: dict) -> dict:
    """Finish a recorded cleanup, including an interruption after preservation."""
    removed = {item["path"]: item for item in summary.get("removed", [])}
    for name in DENSE_EPISODE_DIRS + EPISODE_IMAGE_DIRS:
        size = _remove_path(episode_root / name)
        if size:
            removed[name] = {"path": name, "bytes": size,
                             "kind": "consumed_episode_data"}
    summary["removed"] = list(removed.values())
    summary["removed_bytes"] = sum(item["bytes"] for item in removed.values())
    summary["regenerable"]["removed_dirs"] = list(DENSE_EPISODE_DIRS + EPISODE_IMAGE_DIRS)
    summary["status"] = "completed"
    atomic_write_json(episode_root / "episode_summary.json", summary)
    return summary


def read_episode_summary(episode_root: Path) -> dict:
    """Read an episode's compact summary, or {} if absent/corrupt."""
    path = Path(episode_root) / "episode_summary.json"
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _expected_sample_count(episode_root: Path, meta: dict) -> Optional[int]:
    """Total samples this episode was supposed to synthesise, or None if unknown."""
    for key in ("n_images",):
        value = meta.get(key)
        if isinstance(value, int) and value > 0:
            return value
    views, per_view = meta.get("n_views"), meta.get("samples_per_view")
    if isinstance(views, int) and isinstance(per_view, int) and views > 0 and per_view > 0:
        return views * per_view
    synthesis = episode_root / "stage2_synthesis.json"
    if synthesis.is_file():
        try:
            with open(synthesis, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            prepared = data.get("prepared") or {}
            views, per_view = prepared.get("num_views"), prepared.get("samples_per_view")
            if (data.get("status") == "complete" and isinstance(views, int)
                    and isinstance(per_view, int) and views > 0 and per_view > 0):
                return views * per_view
        except (OSError, json.JSONDecodeError):
            pass
    return None


def _episode_completeness(episode_root: Path, meta: dict, indices: Sequence[int]) -> dict:
    """Check the episode truly finished before anything is deleted.

    An episode is only safe to compact when its synthesis produced the full sample
    set and every retained representative index has all three comparison images.
    Never delete on a partial episode: the originals are the only copy.
    """
    expected = _expected_sample_count(episode_root, meta)
    present = {}
    for directory in EPISODE_IMAGE_DIRS:
        path = episode_root / directory
        if not path.is_dir():
            present[directory] = None
            continue
        names = {int(name.rsplit(".", 1)[0]) for name in os.listdir(path)
                 if name.endswith(".png") and name[:-4].isdigit()}
        present[directory] = names
    missing_dirs = [d for d, names in present.items() if names is None]
    counts = {d: len(names) for d, names in present.items() if names is not None}
    representative_ok = True
    for index in indices:
        for directory in EPISODE_IMAGE_DIRS:
            names = present.get(directory)
            if names is None or index not in names:
                representative_ok = False
    expected_indices = set(range(expected)) if isinstance(expected, int) and expected > 0 else None
    missing_indices = {
        directory: sorted(expected_indices - (names or set()))
        for directory, names in present.items()
    } if expected_indices is not None else {}
    short = {directory: len(missing) for directory, missing in missing_indices.items() if missing}
    complete = expected_indices is not None and bool(indices) and not missing_dirs and representative_ok and not short
    return {
        "expected_samples": expected,
        "present_counts": counts,
        "missing_dirs": missing_dirs,
        "counts_below_expected": short,
        "missing_sample_indices": missing_indices,
        "representative_indices_present": representative_ok,
        "complete": complete,
    }


def finalize_episode(
    stage_dir: os.PathLike | str,
    *,
    episode_dir_name: str,
    episode_idx: int,
    elevation,
    radius,
    iteration_start: Optional[int] = None,
    iteration_end: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    point_cloud_path: Optional[str] = None,
    metrics: Optional[dict] = None,
    timings: Optional[dict] = None,
    source: str = "in_run",
    panel_views: int = PANEL_VIEW_COUNT,
) -> dict:
    """Retain a bounded episode summary, then drop the consumed dense scratch.

    Safe by construction:
      * refuses to delete anything unless the episode's synthesis is complete and
        every retained representative index has all three comparison images;
      * writes the summary atomically BEFORE deleting, so the record of what was
        removed can never be lost by the deletion itself;
      * is idempotent -- an already-compacted episode returns its existing valid
        summary rather than overwriting it with an empty one.
    """
    stage_dir = Path(stage_dir)
    episode_root = stage_dir / "idu" / episode_dir_name
    episode_root.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        name: sha256_file(episode_root / name)
        for name in ("episode_meta.json", "stage2_synthesis.json")
        if (episode_root / name).is_file()
    }
    existing = read_episode_summary(episode_root)
    if (existing.get("status") in ("retained", "completed")
            and existing.get("input_hashes") == input_hashes and existing.get("retained_bytes")):
        kept = [Path(item["path"]) for item in existing.get("retained", [])]
        kept = [path if path.is_absolute() else episode_root / path for path in kept]
        if not kept or not all(path.exists() for path in kept):
            raise RuntimeError(f"Retained episode artifacts are missing: {episode_root}")
        if existing["status"] == "retained":
            return _finish_episode_cleanup(episode_root, existing)
        return {**existing, "idempotent_replay": True}

    meta = {}
    meta_path = episode_root / "episode_meta.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            meta = {}

    available = _sample_indices(episode_root)
    indices = select_panel_indices(available, panel_views)
    completeness = _episode_completeness(episode_root, meta, indices)
    if not completeness["complete"]:
        result = {
            "schema_version": COMPACT_SCHEMA_VERSION, "status": "refused_incomplete",
            "source": source, "updated_at": _now(), "episode_idx": episode_idx,
            "episode_dir": str(episode_root), "elevation": elevation, "radius": radius,
            "completeness": completeness,
            "removed": [], "removed_bytes": 0, "retained_bytes": 0,
            "notes": ["Episode synthesis is not complete; nothing was deleted. The originals are "
                      "the only copy, so compaction is retried only once the full sample set and "
                      "the representative comparison images exist."],
        }
        atomic_write_json(episode_root / "compact_refused.json", result)
        return result

    retained_dir = episode_root / "retained_views"
    panels_dir = episode_root / "panels"
    retained_dir.mkdir(exist_ok=True)

    copied: List[dict] = []
    for index in indices:
        for folder in EPISODE_IMAGE_DIRS:
            source_path = episode_root / folder / f"{index:05d}.png"
            target = retained_dir / f"{index:05d}_{folder}.png"
            shutil.copyfile(source_path, target)
            copied.append({"path": str(target), "bytes": target.stat().st_size,
                           "source": f"{folder}/{index:05d}.png"})

    panel = build_comparison_panel(
        episode_root, indices, panels_dir / f"{episode_dir_name}_panel.jpg",
        title=f"{stage_dir.name} {episode_dir_name} e={elevation} r={radius}",
    )
    if not panel["path"] or panel["missing"]:
        result = {
            "schema_version": COMPACT_SCHEMA_VERSION, "status": "refused_incomplete",
            "source": source, "updated_at": _now(), "episode_idx": episode_idx,
            "episode_dir": str(episode_root), "elevation": elevation, "radius": radius,
            "completeness": completeness, "panel": panel,
            "removed": [], "removed_bytes": 0, "retained_bytes": 0,
            "notes": ["Comparison panel could not be built from complete triples; nothing deleted."],
        }
        atomic_write_json(episode_root / "compact_refused.json", result)
        return result
    compare_html = _compact_compare_html(
        episode_root, indices, EPISODE_IMAGE_DIRS, elevation, radius, episode_idx)

    retained: List[dict] = [
        {"path": name, "bytes": (episode_root / name).stat().st_size}
        for name in KEEP_EPISODE_FILES if (episode_root / name).is_file()
    ]
    for name in KEEP_EPISODE_DIRS:
        if (episode_root / name).is_dir():
            retained.append({"path": name, "bytes": _tree_size(episode_root / name)})
    for item in copied:
        retained.append({"path": item["path"], "bytes": item["bytes"]})
    retained.append({"path": panel["path"], "bytes": panel["bytes"]})
    retained.append({"path": compare_html, "bytes": os.path.getsize(compare_html)})

    prior = sorted((int(p.stem.removeprefix("chkpnt")), p)
                   for p in stage_dir.glob("chkpnt*.pth")
                   if p.stem.removeprefix("chkpnt").isdigit()) if stage_dir.is_dir() else []
    summary = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "status": "retained",
        "source": source,
        "updated_at": _now(),
        "episode_idx": episode_idx,
        "input_hashes": input_hashes,
        "episode_dir": str(episode_root),
        "elevation": elevation,
        "radius": radius,
        "iteration_start": iteration_start,
        "iteration_end": iteration_end,
        "checkpoint": checkpoint_path,
        "point_cloud": point_cloud_path,
        "metrics": metrics or {},
        "timings": timings or {},
        "retained_sample_indices": indices,
        "completeness": completeness,
        "panels": [panel],
        "compare_html": compare_html,
        "retained": retained,
        "removed": [],
        "retained_bytes": sum(item["bytes"] for item in retained),
        "removed_bytes": 0,
        "regenerable": {
            "removed_dirs": list(DENSE_EPISODE_DIRS + EPISODE_IMAGE_DIRS),
            "how": ("Re-run the whole Stage 2 course for this scene from the Stage 1 checkpoint: "
                    "episodes are sequential, so episode N is only reproducible by replaying "
                    "episodes 0..N-1 first (each episode's training state is the next one's "
                    "starting point). Cameras, Qwen3-VL prompts, DLoRAL seeds and MoGE depth are "
                    "then deterministic from the recorded elevation/radius and --idu_seed. The "
                    "per-sample rasters themselves are not stored, so byte-identical outputs are "
                    "not guaranteed across library/driver versions."),
            "not_reproducible_from": ("This episode alone from the Stage 1 checkpoint: prior "
                                      "episodes' training states were needed and have been "
                                      "compacted away."),
        },
        "notes": [
            "Filtered 3D Gaussian PLY and the learned appearance embeddings/MLP live in different "
            "artifacts; both final checkpoint and matching PLY are retained outside this summary.",
            "Retention is durable before deletion; completed status is written only after cleanup.",
        ],
    }
    # Preserve a recoverable transaction before removing any consumed source.
    summary["checkpoints_present_before"] = [p.name for _, p in prior]
    atomic_write_json(episode_root / "episode_summary.json", summary)
    return _finish_episode_cleanup(episode_root, summary)


def stage_episode_summaries(stage_dir: os.PathLike | str) -> List[dict]:
    """Load every per-episode summary in ``<stage_dir>/idu`` (sorted by episode)."""
    idu_dir = Path(stage_dir) / "idu"
    summaries = []
    if not idu_dir.is_dir():
        return summaries
    for path in sorted(idu_dir.glob("episode_*/episode_summary.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                summaries.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue
    summaries.sort(key=lambda item: item.get("episode_idx", 0))
    return summaries


def build_stage_comparison(stage_dir: os.PathLike | str, out_name: str = "episode_comparison.jpg") -> dict:
    """Stack every retained episode panel into one stage-level comparison JPEG."""
    stage_dir = Path(stage_dir)
    panels = []
    for summary in stage_episode_summaries(stage_dir):
        for panel in summary.get("panels", []):
            if panel.get("path") and os.path.isfile(panel["path"]):
                panels.append(panel["path"])
    if not panels:
        return {"path": None, "panels": 0, "bytes": 0}
    images = [Image.open(path).convert("RGB") for path in panels]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in images:
        canvas.paste(image, (0, offset))
        offset += image.height
    out_path = stage_dir / out_name
    canvas.save(out_path, format="JPEG", quality=92)
    return {"path": str(out_path), "panels": len(panels), "bytes": os.path.getsize(out_path)}


def retire_prior_iteration(stage_dir: os.PathLike | str, prior_iteration: int,
                           current_iteration: int, *, protected_below: int) -> dict:
    """Delete the superseded Stage 2 checkpoint/PLY pair.

    Only called after ``chkpnt<current_iteration>.pth`` and
    ``point_cloud/iteration_<current_iteration>/point_cloud.ply`` are known to
    exist.  ``protected_below`` guards the Stage 1 artifacts (a Stage 2 run
    never owns iteration 30000, which is the Stage 1 final pair).
    """
    stage_dir = Path(stage_dir)
    result = {"removed": [], "skipped": [], "bytes": 0}
    if prior_iteration is None or prior_iteration < protected_below:
        result["skipped"].append(f"prior_iteration={prior_iteration} below protection floor "
                                 f"{protected_below}")
        return result
    if prior_iteration >= current_iteration:
        result["skipped"].append(
            f"prior_iteration={prior_iteration} is not older than current_iteration="
            f"{current_iteration}; refusing to retire the just-verified pair")
        return result
    current_checkpoint = stage_dir / f"chkpnt{current_iteration}.pth"
    current_ply = stage_dir / "point_cloud" / f"iteration_{current_iteration}" / "point_cloud.ply"
    for path in (current_checkpoint, current_ply):
        if not path.is_file() or path.stat().st_size == 0:
            result["skipped"].append(f"newer artifact missing/empty: {path}")
            return result
    for path in (stage_dir / f"chkpnt{prior_iteration}.pth",
                 stage_dir / "point_cloud" / f"iteration_{prior_iteration}"):
        size = _remove_path(path)
        if size:
            result["removed"].append(path.name)
            result["bytes"] += size
    return result


#: A TensorBoard event file above this size is a full-resolution image dump
#: (retention.do_not_archive).  Below it, the file holds only scalar/histogram
#: entries -- "compact logs", which the retention list keeps.  A compact-mode run
#: produced ~6-9 KB per episode; the legacy full run produced ~350 MB per episode.
TFEVENT_IMAGE_DUMP_BYTES = 32 * 1024 * 1024


def prune_stage_scratch(stage_dir: os.PathLike | str, *, drop_tfevents: bool = False,
                        keep_tfevents: bool = False) -> dict:
    """Remove stage-level scratch (``depth_tmp``, oversized TB image dumps).

    ``drop_tfevents`` requests the size-aware default; ``keep_tfevents`` forces
    every event file to be retained.  The choice is reported per file so the
    record is explicit about what happened.
    """
    stage_dir = Path(stage_dir)
    removed = []
    kept = []
    size = _remove_path(stage_dir / "depth_tmp")
    if size:
        removed.append({"path": "depth_tmp", "bytes": size, "kind": "moge_scratch"})
    for path in sorted(stage_dir.glob("events.out.tfevents.*")):
        file_size = path.stat().st_size
        if keep_tfevents or file_size < TFEVENT_IMAGE_DUMP_BYTES:
            kept.append({"path": path.name, "bytes": file_size, "kind": "compact_log"})
            continue
        if drop_tfevents:
            removed_size = _remove_path(path)
            if removed_size:
                removed.append({"path": path.name, "bytes": removed_size,
                                "kind": "raw_tensorboard_image_events"})
        else:
            kept.append({"path": path.name, "bytes": file_size,
                         "kind": "oversized_tensorboard_image_events"})
    return {"removed": removed, "kept": kept, "bytes": sum(item["bytes"] for item in removed)}


def compact_stage_output(stage_dir: os.PathLike | str, *, apply: bool = False,
                         panel_views: int = PANEL_VIEW_COUNT, drop_tfevents: bool = True,
                         keep_tfevents: bool = False,
                         drop_intermediate_iterations: bool = True,
                         final_iteration: Optional[int] = None) -> dict:
    """Compact an already-finished stage directory (used for reused runs).

    ``apply=False`` performs a dry run that reports what would be removed.
    """
    stage_dir = Path(stage_dir)
    report: Dict[str, object] = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "stage_dir": str(stage_dir),
        "apply": bool(apply),
        "updated_at": _now(),
        "episodes": [],
        "removed": [],
        "retained": [],
        "notes": [],
    }
    idu_dir = stage_dir / "idu"
    if idu_dir.is_dir():
        for episode_root in sorted(idu_dir.glob("episode_*")):
            if not episode_root.is_dir():
                continue
            existing = episode_root / "episode_summary.json"
            dense_present = any((episode_root / name).is_dir() for name in
                                DENSE_EPISODE_DIRS + EPISODE_IMAGE_DIRS)
            if existing.is_file() and not dense_present:
                report["episodes"].append({"episode": episode_root.name, "action": "already_compact"})
                continue
            if not apply:
                sizes = {name: _tree_size(episode_root / name)
                         for name in DENSE_EPISODE_DIRS + EPISODE_IMAGE_DIRS
                         if (episode_root / name).exists()}
                report["episodes"].append({"episode": episode_root.name, "action": "would_compact",
                                           "removed_bytes": sum(sizes.values()), "sizes": sizes})
                continue
            meta = {}
            meta_path = episode_root / "episode_meta.json"
            if meta_path.is_file():
                try:
                    with open(meta_path, "r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    meta = {}
            summary = finalize_episode(
                stage_dir,
                episode_dir_name=episode_root.name,
                episode_idx=int(meta.get("episode_idx", 0)),
                elevation=meta.get("elevation"),
                radius=meta.get("radius"),
                checkpoint_path=meta.get("checkpoint_used"),
                source="offline_compaction",
                panel_views=panel_views,
            )
            report["episodes"].append({"episode": episode_root.name, "action": "compacted",
                                       "removed_bytes": summary["removed_bytes"],
                                       "retained_bytes": summary["retained_bytes"]})
    comparison = build_stage_comparison(stage_dir) if apply else {"path": None}
    report["comparison"] = comparison

    if drop_intermediate_iterations:
        checkpoints = sorted(
            (int(path.stem.removeprefix("chkpnt")), path)
            for path in stage_dir.glob("chkpnt*.pth")
            if path.stem.removeprefix("chkpnt").isdigit()
        )
        if checkpoints:
            latest = checkpoints[-1][0] if final_iteration is None else final_iteration
            latest_ply = stage_dir / "point_cloud" / f"iteration_{latest}" / "point_cloud.ply"
            if not latest_ply.is_file():
                report["notes"].append(
                    f"final iteration {latest} has no matching PLY; intermediate checkpoint/PLY "
                    "retention left untouched")
            else:
                for iteration, path in checkpoints:
                    if iteration == latest:
                        continue
                    if not apply:
                        size = path.stat().st_size + _tree_size(
                            stage_dir / "point_cloud" / f"iteration_{iteration}")
                        report["removed"].append({"path": path.name, "bytes": size,
                                                  "kind": "intermediate_checkpoint_ply"})
                        continue
                    size = _remove_path(path)
                    size += _remove_path(stage_dir / "point_cloud" / f"iteration_{iteration}")
                    if size:
                        report["removed"].append({"path": path.name, "bytes": size,
                                                  "kind": "intermediate_checkpoint_ply"})

    scratch = prune_stage_scratch(stage_dir, drop_tfevents=drop_tfevents,
                                  keep_tfevents=keep_tfevents) if apply else {
        "removed": [], "kept": [], "bytes": 0}
    if not apply:
        dry_scratch = []
        if (stage_dir / "depth_tmp").exists():
            size = _tree_size(stage_dir / "depth_tmp")
            if size:
                dry_scratch.append({"path": "depth_tmp", "bytes": size, "kind": "moge_scratch"})
        for path in sorted(stage_dir.glob("events.out.tfevents.*")):
            file_size = path.stat().st_size
            if keep_tfevents or file_size < TFEVENT_IMAGE_DUMP_BYTES:
                continue
            if drop_tfevents:
                dry_scratch.append({"path": path.name, "bytes": file_size,
                                    "kind": "raw_tensorboard_image_events"})
        scratch = {"removed": dry_scratch, "bytes": sum(item["bytes"] for item in dry_scratch)}
    report["scratch"] = scratch
    report["removed_bytes"] = sum(item["bytes"] for item in report["removed"]) + scratch["bytes"]
    report["removed_bytes"] += sum(item.get("removed_bytes", 0) for item in report["episodes"])

    retained_names = ["chkpnt*.pth (latest)", "point_cloud/iteration_<latest>/point_cloud.ply",
                      "cfg_args", "cameras.json", "console.log / train.log", "input.ply",
                      "idu/*/episode_summary.json", "idu/*/panels", "idu/*/retained_views",
                      "episode_comparison.jpg"]
    report["retained"] = retained_names
    report["notes"].append(
        "The final checkpoint and its matching filter_3D PLY are retained; only superseded "
        "iterations, consumed episode scratch and raw TensorBoard image events are removed.")
    if apply:
        atomic_write_json(stage_dir / "compact_retention.json", report)
    return report


def compact_stage1_output(stage_dir: os.PathLike | str, *, apply: bool = False,
                          drop_tfevents: bool = True, keep_tfevents: bool = False,
                          keep_iteration: int = 30000) -> dict:
    """Keep only the final Stage 1 checkpoint/PLY (plus small provenance)."""
    stage_dir = Path(stage_dir)
    report = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "stage_dir": str(stage_dir),
        "apply": bool(apply),
        "updated_at": _now(),
        "removed": [],
        "notes": [],
    }
    checkpoint = stage_dir / f"chkpnt{keep_iteration}.pth"
    ply = stage_dir / "point_cloud" / f"iteration_{keep_iteration}" / "point_cloud.ply"
    if not checkpoint.is_file() or not ply.is_file():
        report["notes"].append(
            f"final Stage 1 pair (chkpnt{keep_iteration}.pth + iteration_{keep_iteration}/"
            "point_cloud.ply) is incomplete; nothing removed")
        return report
    for path in sorted(stage_dir.glob("chkpnt*.pth")):
        if path.stem == f"chkpnt{keep_iteration}":
            continue
        if not apply:
            report["removed"].append({"path": path.name, "bytes": path.stat().st_size,
                                      "kind": "intermediate_checkpoint"})
            continue
        size = _remove_path(path)
        if size:
            report["removed"].append({"path": path.name, "bytes": size,
                                      "kind": "intermediate_checkpoint"})
    cloud_root = stage_dir / "point_cloud"
    if cloud_root.is_dir():
        for path in sorted(cloud_root.glob("iteration_*")):
            if path.name == f"iteration_{keep_iteration}":
                continue
            if not apply:
                report["removed"].append({"path": path.name, "bytes": _tree_size(path),
                                          "kind": "intermediate_point_cloud"})
                continue
            size = _remove_path(path)
            if size:
                report["removed"].append({"path": path.name, "bytes": size,
                                          "kind": "intermediate_point_cloud"})
    scratch = prune_stage_scratch(stage_dir, drop_tfevents=drop_tfevents,
                                  keep_tfevents=keep_tfevents) if apply else {
        "removed": [], "kept": [], "bytes": 0}
    if not apply:
        dry_scratch = []
        if (stage_dir / "depth_tmp").exists():
            size = _tree_size(stage_dir / "depth_tmp")
            if size:
                dry_scratch.append({"path": "depth_tmp", "bytes": size, "kind": "moge_scratch"})
        for path in sorted(stage_dir.glob("events.out.tfevents.*")):
            file_size = path.stat().st_size
            if keep_tfevents or file_size < TFEVENT_IMAGE_DUMP_BYTES:
                continue
            if drop_tfevents:
                dry_scratch.append({"path": path.name, "bytes": file_size,
                                    "kind": "raw_tensorboard_image_events"})
        scratch = {"removed": dry_scratch, "bytes": sum(item["bytes"] for item in dry_scratch)}
    report["scratch"] = scratch
    report["removed_bytes"] = sum(item["bytes"] for item in report["removed"]) + scratch["bytes"]
    if apply:
        atomic_write_json(stage_dir / "compact_retention.json", report)
    return report


def sha256_file(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    """SHA256 of a file (used for archive/evaluation provenance)."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
