"""JAX_214 one-ROI pipeline transfer. Not a generalization claim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lod.lineage import POLICY
from lod.panel import FIXED_SETTINGS
from utils.zoom_camera import NormalizedROI


SCENE = "JAX_214"
VIEW_INDEX = 0
VIEW_IMAGE = "JAX_214_018_RGB"
DATASET_DIR = Path("skyfall-gs_exp/skyfall-gs_data/datasets_JAX/JAX_214")
STAGE1_DIR = Path("skyfall-gs_exp/stage1/JAX_214")
RUN_DIR = Path("skyfall-gs_exp/lod_jax214")
STATUS = "second_scene_pipeline"
CLAIM = "jax214_one_roi_pipeline_transfer"

ROI = {
    "id": "building_parking",
    "name": "jax214_building_parking",
    "dominant": "building_parking_junction",
    "center_x": 0.66,
    "center_y": 0.66,
    "width": 0.1,
    "height": 0.1,
    "rationale": (
        "Train view 0 = JAX_214_018_RGB. The box sits on the SE mixed-use "
        "east facade where it meets the large parking lot: building edges on "
        "the north half, vehicle rows on the south half. Water on the west "
        "shore is outside the box. Chosen from GT aiming crops before Stage1 "
        "or absorption metrics. Not a copy of any JAX_068 center."
    ),
}

REJECTED = [
    {
        "id": "b_se_lot_west",
        "center_x": 0.70,
        "center_y": 0.70,
        "reason": "Parking aisles and the access road; almost no building edge.",
    },
    {
        "id": "c_mid_east_lot",
        "center_x": 0.62,
        "center_y": 0.58,
        "reason": "Mostly roofs and a large shadow; parking is not the junction.",
    },
    {
        "id": "d_south_court",
        "center_x": 0.52,
        "center_y": 0.70,
        "reason": "Almost only building mass; parking is a thin strip.",
    },
    {
        "id": "e_north_campus",
        "center_x": 0.58,
        "center_y": 0.42,
        "reason": "Cars on a pale roof/lot, not the SE building-parking junction.",
    },
]

DO_NOT_CLAIM = [
    "cross_scene_generalization",
    "generation_authenticity",
    "geometry_advantage",
    "video_fully_passed",
    "mode_ranking",
]


def panel_roi(item: dict[str, Any] | None = None) -> NormalizedROI:
    item = item or ROI
    return NormalizedROI(
        float(item["center_x"]),
        float(item["center_y"]),
        float(item.get("width", 0.1)),
        float(item.get("height", 0.1)),
    )


def _frames(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Path(frame["file_path"]).name for frame in data["frames"]]


def check_dataset(root: str | Path = DATASET_DIR) -> dict[str, Any]:
    root = Path(root)
    train = _frames(root / "transforms_train.json")
    test = _frames(root / "transforms_test.json")
    images = sorted(p.name for p in (root / "images").glob("*.png"))
    masks = sorted(p.name for p in (root / "masks").glob("*.npy"))
    cameras = sorted(p.stem for p in (root / "cameras").glob("*.json"))
    image_stems = [Path(name).stem for name in images]
    missing_images = [name for name in train + test if name not in images]
    missing_masks = [stem for stem in image_stems if f"{stem}.npy" not in masks]
    missing_cameras = [stem for stem in image_stems if stem not in cameras]
    roi = panel_roi()
    from PIL import Image

    sizes = set()
    for name in train + test:
        with Image.open(root / "images" / name) as image:
            sizes.add(image.size)
    view0 = train[VIEW_INDEX]
    return {
        "scene": SCENE,
        "root": str(root),
        "n_train": len(train),
        "n_test": len(test),
        "n_images": len(images),
        "train": train,
        "test": test,
        "train_test_overlap": sorted(set(train) & set(test)),
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "missing_cameras": missing_cameras,
        "image_sizes": [list(item) for item in sorted(sizes)],
        "points3d_txt": (root / "points3D.txt").is_file(),
        "points3d_ply": (root / "points3D.ply").is_file(),
        "view0": view0,
        "view_index": VIEW_INDEX,
        "view0_matches_lock": Path(view0).stem == VIEW_IMAGE,
        "roi_2x_valid": roi.zoom_is_valid(2.0),
        "roi_4x_valid": roi.zoom_is_valid(4.0),
        "ok": not missing_images
        and not missing_masks
        and not missing_cameras
        and len(train) == 21
        and len(test) == 3
        and not (set(train) & set(test))
        and sizes == {(2048, 2048)}
        and Path(view0).stem == VIEW_IMAGE
        and roi.zoom_is_valid(2.0)
        and roi.zoom_is_valid(4.0),
    }


def selection_payload(*, locked_at: str, dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    roi = panel_roi()
    item = {
        **ROI,
        "min_zoom": roi.min_zoom_factor(),
        "zoom_2x_valid": roi.zoom_is_valid(2.0),
        "zoom_4x_valid": roi.zoom_is_valid(4.0),
    }
    return {
        "schema": "lod_jax214_v1",
        "locked_at": locked_at,
        "locked_before_training": True,
        "claim": CLAIM,
        "status": STATUS,
        "do_not_claim": list(DO_NOT_CLAIM),
        "view": {
            "scene": SCENE,
            "view_index": VIEW_INDEX,
            "image_name": VIEW_IMAGE,
            "source": "locked image_name JAX_214_018_RGB; train-camera list index recorded after Scene load, not used as identity",
        },
        "fixed_settings": FIXED_SETTINGS,
        "policy": POLICY,
        "roi": item,
        "rejected_candidates": REJECTED,
        "not_copied_from": ["JAX_068 panel centers", "jax068_ybuilding", "jax068_0p28_0p28"],
        "water": "west shore of view 0 is outside the locked box; not the ROI subject",
        "dataset": dataset or {"root": str(DATASET_DIR)},
        "notes": [
            "One scene, one ROI. Not a mode ranking and not a generalization claim.",
            "If the chain completes, it is evidence that the same 2x->4x source pipeline ran on a second Stage1.",
            "Absolute MAE is not required to match JAX_068.",
        ],
    }
