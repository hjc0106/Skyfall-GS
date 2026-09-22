"""JAX_068 full-scene eval lock. Test views never enter training or supervision."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lod.camera import camera_image_stem
from lod.eval_protocol import PROTOCOL
from lod.lineage import write_json
from lod.panel import VIEW_IMAGE, VIEW_INDEX


SCENE = "JAX_068"
STAGE1_DIR = Path("skyfall-gs_exp/stage1/JAX_068")
STAGE1_CHECKPOINT = STAGE1_DIR / "chkpnt30000.pth"
DATASET_DIR = Path(
    os.environ.get("JAX068_DATASET_DIR", "data/datasets_JAX/JAX_068")
).expanduser()
EVAL_DIR = Path("skyfall-gs_exp/lod_jax068_eval")
SCENE_DIR = Path("skyfall-gs_exp/lod_jax068_scene")
SCENE_80K_DIR = Path("skyfall-gs_exp/lod_jax068_scene_80k")
TRAIN_VIEW = VIEW_IMAGE  # JAX_068_011_RGB
TEST_VIEWS = ("JAX_068_003_RGB", "JAX_068_012_RGB")

# 2x visible window is 0.5; 50% overlap → centers 0.25 / 0.50 / 0.75, all 4x-legal.
GRID_OVERLAP = 0.5
MIN_MASK_FRAC = 0.70
MIN_WORLD_DIST_FRAC = 0.12
NEIGHBOR_K = 2
ROI_MARKER = 0.1
VOXEL_BINS = 6
PROBE_TILES = 3

# Locked before any joint training. Overlapping; not independent success samples.
TEST_EVAL_WINDOW_CENTERS = (
    ("center", 0.50, 0.50),
    ("nw", 0.30, 0.30),
    ("ne", 0.70, 0.30),
    ("sw", 0.30, 0.70),
    ("se", 0.70, 0.70),
)


def _frames(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [camera_image_stem(frame["file_path"]) for frame in data["frames"]]


def check_dataset(root: str | Path = DATASET_DIR) -> dict[str, Any]:
    root = Path(root)
    train = _frames(root / "transforms_train.json")
    test = _frames(root / "transforms_test.json")
    overlap = sorted(set(train) & set(test))
    return {
        "scene": SCENE,
        "root": str(root),
        "n_train": len(train),
        "n_test": len(test),
        "train": train,
        "test": test,
        "train_view0": train[VIEW_INDEX] if train else None,
        "train_view0_matches_lock": bool(train) and train[VIEW_INDEX] == TRAIN_VIEW,
        "test_matches_lock": tuple(test) == TEST_VIEWS,
        "train_test_overlap": overlap,
        "ok": (
            len(train) == 17
            and tuple(test) == TEST_VIEWS
            and not overlap
            and bool(train)
            and train[VIEW_INDEX] == TRAIN_VIEW
        ),
        "protocol": PROTOCOL,
        "test_images_excluded_from": list(PROTOCOL["test_images_excluded_from"]),
    }


def write_camera_lock(path: str | Path, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dataset or check_dataset()
    write_json(path, payload)
    return payload
