"""Locked local DLoRAL / FlowEdit supervision comparison for JAX_068.

This module is intentionally separate from the nine-center orbit experiment.
It defines the single ``(-128, 0, 0)`` center, six training azimuths, three
supervision arms, and the fixed 4x / 500-step comparison budget.  It does not
select a 54-view or 8x follow-up automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from lod.episode1 import (
    BASE_FOV_DEG,
    CENTER_ROI,
    ELEVATION_DEG,
    IMAGE_SIZE,
    MODEL_STEP_SCALE,
    RADIUS,
)
from lod.jax068 import SCENE, STAGE1_CHECKPOINT
from lod.orbit_9c6a import (
    ALPHA_THRESHOLD,
    AZIMUTHS_DEG,
    DENSIFY_FROM,
    DENSIFY_GRAD_THRESHOLD,
    DENSIFY_INTERVAL,
    DENSIFY_UNTIL,
    DIAGNOSTIC_AZIMUTHS_DEG,
    L1_FEATURE_LR,
    L1_OPACITY_LR,
    L1_POSITION_LR,
    L1_ROTATION_LR,
    L1_SCALING_LR,
    MIX_RATIO,
    UID_BASE_1X,
    UID_BASE_4X,
    UID_BASE_NEIGHBOR,
    UID_DIAG,
    make_orbit_camera,
    pose_id,
    pose_name,
    zoomed_orbit_camera,
)


SCENE_NAME = SCENE
STAGE1_PATH = STAGE1_CHECKPOINT
OUTPUT_ROOT = Path("skyfall-gs_exp/lod_jax068_dloral_flowedit_compare")
REPO_ROOT = Path(__file__).resolve().parents[1]

COMPARE_CENTER = (-128.0, 0.0, 0.0)
COMPARE_AZIMUTHS_DEG = tuple(float(value) for value in AZIMUTHS_DEG)
HELDOUT_AZIMUTHS_DEG = tuple(float(value) for value in DIAGNOSTIC_AZIMUTHS_DEG)
ZOOM_FACTOR = 4.0
N_TARGETS = len(COMPARE_AZIMUTHS_DEG)

COMPARE_STEPS = 500
COMPARE_MAX_POINTS = 50_000
COMPARE_CHECKPOINT_STEPS = (0, 50, 100, 250, 500)
COMPARE_EVAL_STEPS = ",".join(str(value) for value in COMPARE_CHECKPOINT_STEPS)
COMPARE_SEED = 0
COMPARE_MIX_RATIO = MIX_RATIO
COMPARE_DENSIFY_FROM = DENSIFY_FROM
COMPARE_DENSIFY_UNTIL = DENSIFY_UNTIL
COMPARE_DENSIFY_INTERVAL = DENSIFY_INTERVAL
COMPARE_DENSIFY_GRAD_THRESHOLD = DENSIFY_GRAD_THRESHOLD

PROBE_STEPS = 30
PROBE_RESUME_AT = 20
PROBE_EVAL_STEPS = "0,20,30"

FLOWEDIT_MODEL_TYPE = "FLUX"
FLOWEDIT_MODEL_PATH_ENV = "FLOWEDIT_MODEL_PATH"
FLOWEDIT_T_STEPS = 28
FLOWEDIT_N_MIN = 4
FLOWEDIT_N_MAX = 10
FLOWEDIT_N_AVG = 1
FLOWEDIT_SRC_GUIDANCE = 1.5
FLOWEDIT_TAR_GUIDANCE = 5.5
FLOWEDIT_SEED_BASE = 68_000

# The old FlowEdit default mentions a "unique white angular structure".  The
# locked center prompt below follows the existing Qwen inspection for this
# region, which identified a Y-shaped roof structure instead.  Keeping this
# decision in the protocol prevents an accidental prompt change during reruns.
FLOWEDIT_SOURCE_PROMPT = (
    "Aerial satellite image of a complex urban campus with multiple buildings, "
    "parking lots, tree-lined streets, roads, and green spaces. The image is a "
    "low-detail 4x focal crop with soft edges, blur, and warped textures; "
    "preserve the observed layout and the Y-shaped roof structure."
)
FLOWEDIT_TARGET_PROMPT = (
    "Restore this aerial satellite image with faithful preservation of building "
    "silhouettes, roof geometry, mechanical units, parking-lot markings, "
    "tree placement, road edges, and the distinctive Y-shaped roof structure. "
    "Add plausible crisp roof, tree-canopy, and pavement textures only where "
    "supported by the input. Do not invent buildings, windows, floors, roads, "
    "or objects."
)
DLORAL_TARGET_PROMPT = FLOWEDIT_TARGET_PROMPT

# These are fixed image-space boxes.  They are saved in every manifest and
# report so crop comparisons stay paired even when the visible content rotates.
CROP_BOXES = {
    "roof": (0.04, 0.18, 0.46, 0.52),
    "tree_canopy": (0.54, 0.08, 0.96, 0.42),
    "road_edge": (0.42, 0.58, 0.94, 0.94),
}

GROUPS: tuple[dict[str, Any], ...] = (
    {
        "id": "A",
        "name": "dloral",
        "label": "L0 render -> DLoRAL",
        "input_kind": "l0",
        "output_kind": "dloral",
        "shared_dloral": True,
    },
    {
        "id": "B",
        "name": "flowedit",
        "label": "L0 render -> FlowEdit",
        "input_kind": "l0",
        "output_kind": "flowedit_l0",
        "shared_dloral": False,
    },
    {
        "id": "C",
        "name": "dloral_flowedit",
        "label": "L0 render -> DLoRAL -> FlowEdit",
        "input_kind": "dloral",
        "output_kind": "flowedit_dloral",
        "shared_dloral": True,
    },
)


def compare_poses() -> tuple[dict[str, Any], ...]:
    """Return the locked six poses in azimuth order."""

    rows = []
    for index, azimuth in enumerate(COMPARE_AZIMUTHS_DEG):
        rows.append(
            {
                "index": int(index),
                "pose_id": pose_id(COMPARE_CENTER, azimuth),
                "name": pose_name(COMPARE_CENTER, azimuth),
                "lookat": list(COMPARE_CENTER),
                "azimuth_deg": float(azimuth),
                "azimuth_index": int(index),
                "uid_1x": int(UID_BASE_1X + 100 + index),
                "uid_4x": int(UID_BASE_4X + 100 + index),
                "uid_neighbor": int(UID_BASE_NEIGHBOR + 100 + index * 2),
                "is_training_azimuth": True,
            }
        )
    return tuple(rows)


def heldout_poses() -> tuple[dict[str, Any], ...]:
    """Return center-only diagnostic cameras excluded from generation/training."""

    rows = []
    for index, azimuth in enumerate(HELDOUT_AZIMUTHS_DEG):
        rows.append(
            {
                "index": int(index),
                "pose_id": pose_id(COMPARE_CENTER, azimuth),
                "name": pose_name(COMPARE_CENTER, azimuth),
                "lookat": list(COMPARE_CENTER),
                "azimuth_deg": float(azimuth),
                "uid_1x": int(UID_BASE_1X + 300 + index),
                "uid_4x": int(UID_BASE_4X + 300 + index),
                "is_training_azimuth": False,
            }
        )
    return tuple(rows)


def flowedit_seed(azimuth_deg: float) -> int:
    """Deterministic per-view seed shared by B and C."""

    index = min(
        range(len(COMPARE_AZIMUTHS_DEG)),
        key=lambda item: abs(COMPARE_AZIMUTHS_DEG[item] - float(azimuth_deg)),
    )
    if abs(COMPARE_AZIMUTHS_DEG[index] - float(azimuth_deg)) > 1e-6:
        raise ValueError(f"{azimuth_deg} is not one of the six locked training azimuths")
    return int(FLOWEDIT_SEED_BASE + index)


def compare_protocol(
    *,
    output_root: str | Path = OUTPUT_ROOT,
    stage1_checkpoint: str | Path = STAGE1_PATH,
    flowedit_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the human-readable protocol written before generation."""

    return {
        "schema": "jax068_single_center_dloral_flowedit_compare_v1",
        "scene": SCENE_NAME,
        "output_root": str(Path(output_root)),
        "stage1_checkpoint": str(Path(stage1_checkpoint)),
        "center": list(COMPARE_CENTER),
        "camera": {
            "azimuths_deg": list(COMPARE_AZIMUTHS_DEG),
            "heldout_azimuths_deg": list(HELDOUT_AZIMUTHS_DEG),
            "elevation_deg": float(ELEVATION_DEG),
            "radius": float(RADIUS),
            "base_fov_deg": float(BASE_FOV_DEG),
            "zoom_factor": float(ZOOM_FACTOR),
            "image_size": [int(IMAGE_SIZE), int(IMAGE_SIZE)],
            "roi": {
                "center_x": float(CENTER_ROI.center_x),
                "center_y": float(CENTER_ROI.center_y),
                "width": float(CENTER_ROI.width),
                "height": float(CENTER_ROI.height),
            },
        },
        "groups": [dict(group) for group in GROUPS],
        "supervision": {
            "l0_shared": True,
            "dloral_shared_by": ["A", "C"],
            "flowedit_prompt_shared_by": ["B", "C"],
            "flowedit_model_type": FLOWEDIT_MODEL_TYPE,
            "flowedit_model_path": None if flowedit_model_path is None else str(flowedit_model_path),
            "T_steps": int(FLOWEDIT_T_STEPS),
            "n_min": int(FLOWEDIT_N_MIN),
            "n_max": int(FLOWEDIT_N_MAX),
            "n_avg": int(FLOWEDIT_N_AVG),
            "src_guidance_scale": float(FLOWEDIT_SRC_GUIDANCE),
            "tar_guidance_scale": float(FLOWEDIT_TAR_GUIDANCE),
            "flowedit_seed_base": int(FLOWEDIT_SEED_BASE),
            "flowedit_seed_by_azimuth": {
                str(int(azimuth)): flowedit_seed(azimuth) for azimuth in COMPARE_AZIMUTHS_DEG
            },
            "flowedit_resets_rng_before_each_image": True,
            "training_rng_isolated": True,
            "dloral_alignment": "geometry",
            "dloral_seed": int(COMPARE_SEED),
        },
        "prompts": {
            "flowedit_source_prompt": FLOWEDIT_SOURCE_PROMPT,
            "flowedit_target_prompt": FLOWEDIT_TARGET_PROMPT,
            "dloral_target_prompt": DLORAL_TARGET_PROMPT,
            "legacy_white_angular_structure": {
                "phrase": "white angular structure",
                "status": "not_used",
                "reason": (
                    "The existing center prompt inspection describes a Y-shaped roof "
                    "structure; the legacy default phrase is not supported for this center."
                ),
            },
        },
        "crops": {name: list(box) for name, box in CROP_BOXES.items()},
        "training": {
            "initial_model": "same Stage1 L0 plus empty L1",
            "trainable": "L1 only",
            "steps": int(COMPARE_STEPS),
            "max_l1_points": int(COMPARE_MAX_POINTS),
            "mix_ratio": float(COMPARE_MIX_RATIO),
            "seed": int(COMPARE_SEED),
            "step_scale": float(MODEL_STEP_SCALE),
            "densify_from": int(COMPARE_DENSIFY_FROM),
            "densify_until": int(COMPARE_DENSIFY_UNTIL),
            "densify_interval": int(COMPARE_DENSIFY_INTERVAL),
            "densify_grad_threshold": float(COMPARE_DENSIFY_GRAD_THRESHOLD),
            "checkpoint_steps": list(COMPARE_CHECKPOINT_STEPS),
            "same_sampling_sequence": True,
            "probe_steps": int(PROBE_STEPS),
            "probe_resume_at": int(PROBE_RESUME_AT),
            "probe_is_not_formal_start": True,
        },
        "evaluation": {
            "supervision_cross_view": "fixed initial L0 camera-Z and common parent mask",
            "training_cross_view": "fixed initial L0 camera-Z and common parent mask",
            "heldout_cross_view": "center-only diagnostic azimuths; no generation or training",
            "old_scale": "native 1x versus initial L0 and all 17 original train photographs",
            "absorption_warning": (
                "Each arm has a different pseudo-label; own-label L1 is descriptive, "
                "not a standalone winner criterion."
            ),
            "automatic_followup": False,
            "eight_x": "blocked pending manual review",
            "fifty_four_view": "blocked pending manual review",
        },
        "notes": [
            "Previous unseeded previews remain in their original archive and are not reused as formal samples.",
            "This local comparison does not modify the nine-center/full54 archives.",
            "Test views JAX_068_003_RGB and JAX_068_012_RGB are outside this local mode-selection experiment.",
            "A and C must reference one identical DLoRAL artifact per azimuth.",
        ],
    }


__all__ = [
    "ALPHA_THRESHOLD",
    "BASE_FOV_DEG",
    "COMPARE_AZIMUTHS_DEG",
    "COMPARE_CENTER",
    "COMPARE_CHECKPOINT_STEPS",
    "COMPARE_DENSIFY_FROM",
    "COMPARE_DENSIFY_GRAD_THRESHOLD",
    "COMPARE_DENSIFY_INTERVAL",
    "COMPARE_DENSIFY_UNTIL",
    "COMPARE_EVAL_STEPS",
    "COMPARE_MAX_POINTS",
    "COMPARE_MIX_RATIO",
    "COMPARE_SEED",
    "COMPARE_STEPS",
    "CROP_BOXES",
    "DLORAL_TARGET_PROMPT",
    "FLOWEDIT_MODEL_TYPE",
    "FLOWEDIT_N_AVG",
    "FLOWEDIT_N_MAX",
    "FLOWEDIT_N_MIN",
    "FLOWEDIT_SEED_BASE",
    "FLOWEDIT_SOURCE_PROMPT",
    "FLOWEDIT_SRC_GUIDANCE",
    "FLOWEDIT_TAR_GUIDANCE",
    "FLOWEDIT_T_STEPS",
    "GROUPS",
    "HELDOUT_AZIMUTHS_DEG",
    "IMAGE_SIZE",
    "MODEL_STEP_SCALE",
    "N_TARGETS",
    "OUTPUT_ROOT",
    "PROBE_EVAL_STEPS",
    "PROBE_RESUME_AT",
    "PROBE_STEPS",
    "STAGE1_PATH",
    "compare_poses",
    "compare_protocol",
    "flowedit_seed",
    "heldout_poses",
    "make_orbit_camera",
    "pose_id",
    "pose_name",
    "zoomed_orbit_camera",
]
