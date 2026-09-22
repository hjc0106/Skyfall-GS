"""JAX_068 L0-30K start, freeze L0, five-episode shared L1, 108 views each.

This protocol is independent of the finished six-view L1-40K continuation.
Training densify/LR/Adam rules are copied from the original 2x -> L1 entry
``scripts/run_lod_jax068_c_two_scale.py`` and run on a cumulative step clock.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from lod.orbit_9c6a import (
    AZIMUTHS_DEG,
    CENTER_ROI,
    DENSIFY_FROM,
    DENSIFY_GRAD_THRESHOLD,
    DENSIFY_INTERVAL,
    DIAGNOSTIC_AZIMUTHS_DEG,
    IMAGE_SIZE,
    L1_FEATURE_LR,
    L1_OPACITY_LR,
    L1_POSITION_LR,
    L1_ROTATION_LR,
    L1_SCALING_LR,
    MODEL_STEP_SCALE,
    PREFERRED_NEIGHBOR_DELTAS_DEG,
    STAGE1_PATH,
    lookats,
    make_orbit_camera,
    neighbor_azimuth,
    pose_id as orbit_pose_id,
    pose_index as orbit_pose_index,
    zoomed_orbit_camera,
)


SCENE = "JAX_068"
SCHEMA = "jax068_l0_start_108_five_episode_v1"
TWO_SCALE_ENTRY = "scripts/run_lod_jax068_c_two_scale.py"
ARCHIVE_NAME = "lod_jax068_c_episodes_l0_start_108"
PREVIOUS_L1_CONTINUATION_ARCHIVE = "lod_jax068_c_episodes_l1"

N_EPISODES = 5
N_LOOKATS = 9
N_AZIMUTHS = 6
N_POSES = N_LOOKATS * N_AZIMUTHS  # 54
SAMPLES_PER_POSE = 2
N_SUPERVISION = N_POSES * SAMPLES_PER_POSE  # 108
STEPS_PER_EPISODE = 5_000
CUMULATIVE_STEPS = N_EPISODES * STEPS_PER_EPISODE  # 25_000
ZOOM_FACTOR = 2.0
SCALE_ZOOM_FACTORS = (1.0, 2.0, 4.0)
BASE_FOV_DEG = 60.0
TRAINED_LEVEL = 1
MAX_POINTS_L1 = 450_000
MIX_RATIO = 0.0
# Copied from ``run_lod_jax068_c_two_scale.py --densify_until_l1`` default.
# The two-scale L1 budget was 40k; densify stopped at 20k.  This run uses the
# same cutoff on the cumulative 25k clock, not a per-episode restart.
DENSIFY_UNTIL = 20_000
CHECKPOINT_STEPS = (0, 250, 500, 1_000, 2_000, 5_000)
EXPECTED_VISITS_PER_IMAGE = STEPS_PER_EPISODE / N_SUPERVISION  # ~46.296
GENERATION_SEED = 4001
TRAINING_SEED = 0
EMPTY_L1_RGB_L1 = 1e-6

EPISODE_SPECS = (
    {"index": 1, "elevation_deg": 85.0, "radius": 300.0},
    {"index": 2, "elevation_deg": 75.0, "radius": 275.0},
    {"index": 3, "elevation_deg": 65.0, "radius": 250.0},
    {"index": 4, "elevation_deg": 55.0, "radius": 225.0},
    {"index": 5, "elevation_deg": 45.0, "radius": 200.0},
)

UID_BASE_1X = 600_000
UID_BASE_2X = 610_000
UID_BASE_NEIGHBOR = 620_000
UID_BASE_HELDOUT = 630_000
UID_STRIDE = 2_000

SAMPLE_IDS = (0, 1)

# Keep DLoRAL and FlowEdit on isolated seeds; samples of one pose differ.
FLOWEDIT_SEED_OFFSET = 500_000


def _num_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def episode_spec(index: int) -> dict[str, Any]:
    spec = next((item for item in EPISODE_SPECS if int(item["index"]) == int(index)), None)
    if spec is None:
        raise ValueError(f"episode index must be 1..{N_EPISODES}, got {index}")
    return dict(spec)


def episode_ids() -> tuple[int, ...]:
    return tuple(int(item["index"]) for item in EPISODE_SPECS)


def lookat_centers() -> tuple[tuple[float, float, float], ...]:
    centers = lookats()
    if len(centers) != N_LOOKATS:
        raise RuntimeError(f"expected {N_LOOKATS} lookats, got {len(centers)}")
    if any(abs(float(item[2])) > 1e-9 for item in centers):
        raise RuntimeError("locked lookats must have Z=0")
    return centers


def pose_id(lookat: Sequence[float], azimuth_deg: float) -> str:
    return orbit_pose_id(lookat, azimuth_deg)


def supervision_id(*, episode: int, pose_id_value: str, sample_id: int) -> str:
    if int(sample_id) not in SAMPLE_IDS:
        raise ValueError(f"sample_id must be 0 or 1, got {sample_id}")
    return f"e{int(episode)}_{pose_id_value}__s{int(sample_id)}"


def _uid(episode: int, kind: str, offset: int) -> int:
    bases = {
        "1x": UID_BASE_1X,
        "2x": UID_BASE_2X,
        "neighbor": UID_BASE_NEIGHBOR,
        "heldout": UID_BASE_HELDOUT,
    }
    return int(bases[kind] + (int(episode) - 1) * UID_STRIDE + offset)


def camera_name(item: Mapping[str, Any], *, zoom: bool = False) -> str:
    suffix = f"_zoom{float(ZOOM_FACTOR):g}" if zoom else ""
    return (
        f"e{int(item['episode'])}_{item['pose_id']}"
        f"_e{float(item['elevation_deg']):g}_r{float(item['radius']):g}{suffix}"
    )


def camera_records(episode: int) -> tuple[dict[str, Any], ...]:
    spec = episode_spec(episode)
    rows = []
    for lookat in lookat_centers():
        for azimuth in AZIMUTHS_DEG:
            index = orbit_pose_index(lookat, azimuth)
            pid = pose_id(lookat, azimuth)
            rows.append(
                {
                    "episode": int(spec["index"]),
                    "pose_index": int(index),
                    "pose_id": pid,
                    "lookat": [float(lookat[0]), float(lookat[1]), float(lookat[2])],
                    "lookat_index": index // N_AZIMUTHS,
                    "azimuth_deg": float(azimuth),
                    "azimuth_index": index % N_AZIMUTHS,
                    "elevation_deg": float(spec["elevation_deg"]),
                    "radius": float(spec["radius"]),
                    "uid_1x": _uid(spec["index"], "1x", index),
                    "uid_2x": _uid(spec["index"], "2x", index),
                    "uid_neighbor": _uid(spec["index"], "neighbor", index * 2),
                    "zoom_factor": float(ZOOM_FACTOR),
                    "image_size": [int(IMAGE_SIZE), int(IMAGE_SIZE)],
                }
            )
    if len(rows) != N_POSES:
        raise RuntimeError(f"expected {N_POSES} cameras, got {len(rows)}")
    return tuple(rows)


def spawn_camera_records() -> tuple[dict[str, Any], ...]:
    """L1 is spawned once from Episode 1's 54 1x cameras at 2x."""

    return camera_records(1)


def training_items(episode: int) -> tuple[dict[str, Any], ...]:
    rows = []
    for rec in camera_records(episode):
        for sample_id in SAMPLE_IDS:
            item = dict(rec)
            item["sample_id"] = int(sample_id)
            item["supervision_id"] = supervision_id(
                episode=int(rec["episode"]),
                pose_id_value=str(rec["pose_id"]),
                sample_id=int(sample_id),
            )
            item["dloral_seed"], item["flowedit_seed"] = sample_seeds(
                generation_seed=GENERATION_SEED,
                episode=int(rec["episode"]),
                pose_index=int(rec["pose_index"]),
                sample_id=int(sample_id),
            )
            rows.append(item)
    if len(rows) != N_SUPERVISION:
        raise RuntimeError(f"expected {N_SUPERVISION} supervision items, got {len(rows)}")
    ids = [item["supervision_id"] for item in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("supervision IDs are not unique")
    return tuple(rows)


def heldout_records(episode: int) -> tuple[dict[str, Any], ...]:
    spec = episode_spec(episode)
    rows = []
    offset = 0
    for lookat in lookat_centers():
        for azimuth in DIAGNOSTIC_AZIMUTHS_DEG:
            rows.append(
                {
                    "episode": int(spec["index"]),
                    "pose_id": f"heldout_{pose_id(lookat, azimuth)}",
                    "lookat": [float(lookat[0]), float(lookat[1]), float(lookat[2])],
                    "azimuth_deg": float(azimuth),
                    "elevation_deg": float(spec["elevation_deg"]),
                    "radius": float(spec["radius"]),
                    "uid_1x": _uid(spec["index"], "heldout", offset),
                    "uid_2x": _uid(spec["index"], "heldout", offset) + 1_000,
                    "zoom_factor": float(ZOOM_FACTOR),
                    "held_out": True,
                }
            )
            offset += 1
    return tuple(rows)


def neighbor_azimuths(azimuth_deg: float) -> tuple[float, ...]:
    return tuple(neighbor_azimuth(azimuth_deg, delta) for delta in PREFERRED_NEIGHBOR_DELTAS_DEG)


def sample_seeds(
    *,
    generation_seed: int,
    episode: int,
    pose_index: int,
    sample_id: int,
) -> tuple[int, int]:
    if int(sample_id) not in SAMPLE_IDS:
        raise ValueError(f"sample_id must be 0 or 1, got {sample_id}")
    dloral = (
        int(generation_seed) * 1_000_000
        + int(episode) * 10_000
        + int(pose_index) * 10
        + int(sample_id)
    )
    return int(dloral), int(dloral + FLOWEDIT_SEED_OFFSET)


def cumulative_step(episode: int, local_step: int) -> int:
    if int(episode) < 1 or int(episode) > N_EPISODES:
        raise ValueError(f"episode must be 1..{N_EPISODES}")
    if int(local_step) < 0 or int(local_step) > STEPS_PER_EPISODE:
        raise ValueError(f"local_step must be 0..{STEPS_PER_EPISODE}")
    return (int(episode) - 1) * STEPS_PER_EPISODE + int(local_step)


def densify_at_cumulative(step: int) -> bool:
    value = int(step)
    return (
        int(DENSIFY_FROM) <= value <= int(DENSIFY_UNTIL)
        and value % int(DENSIFY_INTERVAL) == 0
    )


def pools_disjoint(previous_ids: Sequence[str], current_ids: Sequence[str]) -> bool:
    return not set(previous_ids) & set(current_ids)


def parent_hash_mismatch(saved_sha256: str | None, current_sha256: str) -> bool:
    return str(saved_sha256 or "") != str(current_sha256)


def base_camera(item: Mapping[str, Any], *, data_device: str = "cuda"):
    return make_orbit_camera(
        item["lookat"],
        float(item["azimuth_deg"]),
        uid=int(item["uid_1x"]),
        data_device=data_device,
        elevation_deg=float(item["elevation_deg"]),
        radius=float(item["radius"]),
        image_name=camera_name(item, zoom=False),
    )


def zoom_camera(item: Mapping[str, Any], *, data_device: str = "cuda"):
    return zoomed_orbit_camera(
        base_camera(item, data_device=data_device),
        float(item.get("zoom_factor", ZOOM_FACTOR)),
        uid=int(item["uid_2x"]),
    )


def neighbor_camera(
    item: Mapping[str, Any],
    azimuth_deg: float,
    *,
    uid: int,
    data_device: str = "cuda",
):
    return make_orbit_camera(
        item["lookat"],
        float(azimuth_deg),
        uid=int(uid),
        data_device=data_device,
        elevation_deg=float(item["elevation_deg"]),
        radius=float(item["radius"]),
        image_name=(
            f"e{int(item['episode'])}_neighbor_{pose_id(item['lookat'], azimuth_deg)}"
            f"_e{float(item['elevation_deg']):g}_r{float(item['radius']):g}"
        ),
    )


def two_scale_l1_training_rules() -> dict[str, Any]:
    return {
        "source_entry": TWO_SCALE_ENTRY,
        "not_source": [
            "scripts/run_lod_jax068_c_episodes.py",
            "skyfall-gs_exp/lod_jax068_c_episodes_l1",
            "0.1x continuation",
        ],
        "mix_ratio": float(MIX_RATIO),
        "original_photo_mix": 0.0,
        "previous_episode_mix": 0.0,
        "sampling": "current_episode_108_restored_images_only",
        "loss": "rgb_l1",
        "trained_level": TRAINED_LEVEL,
        "trainable": ["xyz", "sh", "log_scales", "rotations", "opacity_logits"],
        "frozen": ["L0", "appearance_mlp", "appearance_embeddings"],
        "max_points_l1": int(MAX_POINTS_L1),
        "learning_rate": {
            "xyz": float(L1_POSITION_LR),
            "sh": float(L1_FEATURE_LR),
            "opacity_logits": float(L1_OPACITY_LR),
            "log_scales": float(L1_SCALING_LR),
            "rotations": float(L1_ROTATION_LR),
            "schedule": "constant",
            "scale": 1.0,
        },
        "densify": {
            "clock": "cumulative_step",
            "from": int(DENSIFY_FROM),
            "until": int(DENSIFY_UNTIL),
            "interval": int(DENSIFY_INTERVAL),
            "grad_threshold": float(DENSIFY_GRAD_THRESHOLD),
            "reset_on_episode_boundary": False,
            "stage_cameras": "episode1_54_2x_spawn_set",
        },
        "optimizer_policy": "keep_adam_across_episodes",
        "seed": int(TRAINING_SEED),
        "steps_per_episode": int(STEPS_PER_EPISODE),
        "cumulative_steps": int(CUMULATIVE_STEPS),
        "checkpoint_steps_local": list(CHECKPOINT_STEPS),
        "expected_visits_per_image": float(EXPECTED_VISITS_PER_IMAGE),
        "step_scale": float(MODEL_STEP_SCALE),
        "zoom_factor": float(ZOOM_FACTOR),
        "center_roi": {
            "center_x": float(CENTER_ROI.center_x),
            "center_y": float(CENTER_ROI.center_y),
            "width": float(CENTER_ROI.width),
            "height": float(CENTER_ROI.height),
        },
    }


def protocol_payload() -> dict[str, Any]:
    items = training_items(1)
    return {
        "schema": SCHEMA,
        "scene": SCENE,
        "archive": ARCHIVE_NAME,
        "start": "stage1_L0_30k",
        "stage1_checkpoint": str(STAGE1_PATH),
        "comparison_archives": {
            "original_l0": str(STAGE1_PATH),
            "this_run": ARCHIVE_NAME,
            "previous_five_episode_from_old_l1_40k": PREVIOUS_L1_CONTINUATION_ARCHIVE,
        },
        "episodes": [dict(item) for item in EPISODE_SPECS],
        "camera": {
            "lookats": [list(item) for item in lookat_centers()],
            "azimuths_deg": list(AZIMUTHS_DEG),
            "heldout_azimuths_deg": list(DIAGNOSTIC_AZIMUTHS_DEG),
            "n_centers": N_LOOKATS,
            "n_azimuths": N_AZIMUTHS,
            "n_poses": N_POSES,
            "samples_per_pose": SAMPLES_PER_POSE,
            "n_supervision": N_SUPERVISION,
            "base_fov_deg": float(BASE_FOV_DEG),
            "zoom_factor": float(ZOOM_FACTOR),
            "image_size": [int(IMAGE_SIZE), int(IMAGE_SIZE)],
            "neighbor_deltas_deg": list(PREFERRED_NEIGHBOR_DELTAS_DEG),
            "focal_zoom_not_pixel_upsample": True,
        },
        "generation": {
            "order": [
                "parent_render_54_2x_neighbor_depth_1x_context",
                "one_vlm_prompt_per_pose",
                "two_samples_dloral_then_flowedit",
            ],
            "shared_per_pose": ["render_input", "wide_1x", "neighbor", "geometry", "prompt"],
            "per_sample": ["dloral_seed", "flowedit_seed", "supervision_id"],
            "resume": "reuse_valid_dloral_then_flowedit",
            "parent_hash_mismatch": "refuse",
            "generation_seed": int(GENERATION_SEED),
            "sample_ids": list(SAMPLE_IDS),
            "example_ids": [item["supervision_id"] for item in items[:4]],
        },
        "training": two_scale_l1_training_rules(),
        "episode1": "L0_only_supervision_empty_L1_spawn_and_densify",
        "episode2_to_5": "previous_episode_L0_plus_L1_parent_continue_same_L1",
        "probe_is_not_formal_start": True,
        "test_views_not_for_step_selection": ["JAX_068_003_RGB", "JAX_068_012_RGB"],
        "do_not_attribute_all_gaps_to_removing_old_l1_noise": True,
    }


__all__ = [
    "ARCHIVE_NAME",
    "AZIMUTHS_DEG",
    "CHECKPOINT_STEPS",
    "CUMULATIVE_STEPS",
    "DENSIFY_FROM",
    "DENSIFY_INTERVAL",
    "DENSIFY_UNTIL",
    "EPISODE_SPECS",
    "EXPECTED_VISITS_PER_IMAGE",
    "GENERATION_SEED",
    "IMAGE_SIZE",
    "MAX_POINTS_L1",
    "N_EPISODES",
    "N_POSES",
    "N_SUPERVISION",
    "SAMPLES_PER_POSE",
    "SCHEMA",
    "STAGE1_PATH",
    "STEPS_PER_EPISODE",
    "TRAINING_SEED",
    "TWO_SCALE_ENTRY",
    "ZOOM_FACTOR",
    "base_camera",
    "camera_records",
    "cumulative_step",
    "densify_at_cumulative",
    "episode_spec",
    "heldout_records",
    "neighbor_azimuths",
    "neighbor_camera",
    "parent_hash_mismatch",
    "pools_disjoint",
    "protocol_payload",
    "sample_seeds",
    "spawn_camera_records",
    "supervision_id",
    "training_items",
    "two_scale_l1_training_rules",
    "zoom_camera",
]
