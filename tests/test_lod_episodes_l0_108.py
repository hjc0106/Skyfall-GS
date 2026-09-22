from __future__ import annotations

import unittest

from lod.episodes_l0_108 import (
    ARCHIVE_NAME,
    AZIMUTHS_DEG,
    CUMULATIVE_STEPS,
    DENSIFY_FROM,
    DENSIFY_INTERVAL,
    DENSIFY_UNTIL,
    EPISODE_SPECS,
    EXPECTED_VISITS_PER_IMAGE,
    IMAGE_SIZE,
    MAX_POINTS_L1,
    MIX_RATIO,
    N_EPISODES,
    N_POSES,
    N_SUPERVISION,
    PREVIOUS_L1_CONTINUATION_ARCHIVE,
    SAMPLES_PER_POSE,
    STEPS_PER_EPISODE,
    TWO_SCALE_ENTRY,
    ZOOM_FACTOR,
    base_camera,
    camera_records,
    cumulative_step,
    densify_at_cumulative,
    episode_spec,
    heldout_records,
    lookat_centers,
    parent_hash_mismatch,
    pools_disjoint,
    protocol_payload,
    sample_seeds,
    spawn_camera_records,
    training_items,
    two_scale_l1_training_rules,
    zoom_camera,
)
from lod.orbit_9c6a import (
    DENSIFY_FROM as TWO_SCALE_DENSIFY_FROM,
    DENSIFY_GRAD_THRESHOLD,
    DENSIFY_INTERVAL as TWO_SCALE_DENSIFY_INTERVAL,
    L1_FEATURE_LR,
    L1_OPACITY_LR,
    L1_POSITION_LR,
    L1_ROTATION_LR,
    L1_SCALING_LR,
)


class L0Start108ProtocolTests(unittest.TestCase):
    def test_locked_table(self):
        self.assertEqual(N_EPISODES, 5)
        self.assertEqual(N_POSES, 54)
        self.assertEqual(SAMPLES_PER_POSE, 2)
        self.assertEqual(N_SUPERVISION, 108)
        self.assertEqual(STEPS_PER_EPISODE, 5_000)
        self.assertEqual(ZOOM_FACTOR, 2.0)
        self.assertEqual(IMAGE_SIZE, 2048)
        self.assertEqual(
            [(item["index"], item["elevation_deg"], item["radius"]) for item in EPISODE_SPECS],
            [
                (1, 85.0, 300.0),
                (2, 75.0, 275.0),
                (3, 65.0, 250.0),
                (4, 55.0, 225.0),
                (5, 45.0, 200.0),
            ],
        )
        centers = lookat_centers()
        self.assertEqual(len(centers), 9)
        self.assertEqual(sorted({(x, y, z) for x, y, z in centers}), sorted(
            (float(x), float(y), 0.0) for x in (-128.0, 0.0, 128.0) for y in (-128.0, 0.0, 128.0)
        ))
        self.assertEqual(AZIMUTHS_DEG, (0.0, 60.0, 120.0, 180.0, 240.0, 300.0))

    def test_108_unique_ids_and_dual_sample_cameras(self):
        items = training_items(1)
        self.assertEqual(len(items), 108)
        ids = [item["supervision_id"] for item in items]
        self.assertEqual(len(set(ids)), 108)
        self.assertTrue(all("__s0" in item or "__s1" in item for item in ids))
        by_pose = {}
        for item in items:
            by_pose.setdefault(item["pose_id"], []).append(item)
        self.assertEqual(len(by_pose), 54)
        for pose_id, pair in by_pose.items():
            self.assertEqual(len(pair), 2)
            left, right = pair
            self.assertEqual(left["uid_1x"], right["uid_1x"])
            self.assertEqual(left["uid_2x"], right["uid_2x"])
            self.assertEqual(left["azimuth_deg"], right["azimuth_deg"])
            self.assertEqual(left["lookat"], right["lookat"])
            self.assertNotEqual(left["dloral_seed"], right["dloral_seed"])
            self.assertNotEqual(left["flowedit_seed"], right["flowedit_seed"])
            self.assertNotEqual(left["supervision_id"], right["supervision_id"])
            self.assertIn(pose_id, left["supervision_id"])
            self.assertIn("e1_", left["supervision_id"])

    def test_two_scale_l1_rules_not_continuation(self):
        rules = two_scale_l1_training_rules()
        self.assertEqual(rules["source_entry"], TWO_SCALE_ENTRY)
        self.assertEqual(rules["mix_ratio"], 0.0)
        self.assertEqual(rules["original_photo_mix"], 0.0)
        self.assertEqual(rules["previous_episode_mix"], 0.0)
        self.assertEqual(rules["learning_rate"]["xyz"], L1_POSITION_LR)
        self.assertEqual(rules["learning_rate"]["sh"], L1_FEATURE_LR)
        self.assertEqual(rules["learning_rate"]["opacity_logits"], L1_OPACITY_LR)
        self.assertEqual(rules["learning_rate"]["log_scales"], L1_SCALING_LR)
        self.assertEqual(rules["learning_rate"]["rotations"], L1_ROTATION_LR)
        self.assertEqual(rules["learning_rate"]["scale"], 1.0)
        self.assertEqual(rules["densify"]["from"], TWO_SCALE_DENSIFY_FROM)
        self.assertEqual(rules["densify"]["interval"], TWO_SCALE_DENSIFY_INTERVAL)
        self.assertEqual(rules["densify"]["grad_threshold"], DENSIFY_GRAD_THRESHOLD)
        self.assertEqual(rules["densify"]["until"], 20_000)
        self.assertFalse(rules["densify"]["reset_on_episode_boundary"])
        self.assertEqual(rules["optimizer_policy"], "keep_adam_across_episodes")
        self.assertEqual(rules["max_points_l1"], 450_000)
        self.assertEqual(MAX_POINTS_L1, 450_000)
        self.assertEqual(MIX_RATIO, 0.0)
        self.assertNotIn("0.1", str(rules["learning_rate"]))
        payload = protocol_payload()
        self.assertEqual(payload["archive"], ARCHIVE_NAME)
        self.assertEqual(
            payload["comparison_archives"]["previous_five_episode_from_old_l1_40k"],
            PREVIOUS_L1_CONTINUATION_ARCHIVE,
        )
        self.assertNotEqual(ARCHIVE_NAME, PREVIOUS_L1_CONTINUATION_ARCHIVE)
        self.assertTrue(payload["probe_is_not_formal_start"])

    def test_cumulative_densify_clock(self):
        self.assertEqual(cumulative_step(1, 0), 0)
        self.assertEqual(cumulative_step(1, 5_000), 5_000)
        self.assertEqual(cumulative_step(5, 5_000), CUMULATIVE_STEPS)
        self.assertEqual(DENSIFY_FROM, 1)
        self.assertEqual(DENSIFY_INTERVAL, 10)
        self.assertEqual(DENSIFY_UNTIL, 20_000)
        self.assertTrue(densify_at_cumulative(10))
        self.assertTrue(densify_at_cumulative(20_000))
        self.assertFalse(densify_at_cumulative(0))
        self.assertFalse(densify_at_cumulative(20_010))
        self.assertFalse(densify_at_cumulative(25_000))
        # Episode 5 starts at 20k, so densify does not restart.
        self.assertFalse(densify_at_cumulative(cumulative_step(5, 10)))

    def test_episode_switch_replaces_pool(self):
        first = [item["supervision_id"] for item in training_items(1)]
        second = [item["supervision_id"] for item in training_items(2)]
        self.assertTrue(pools_disjoint(first, second))
        visits = {item_id: 0 for item_id in first}
        self.assertEqual(sum(visits.values()), 0)
        self.assertEqual(len(heldout_records(1)), 54)
        heldout_az = {float(item["azimuth_deg"]) for item in heldout_records(1)}
        self.assertTrue(heldout_az.isdisjoint(set(AZIMUTHS_DEG)))

    def test_parent_hash_gate(self):
        self.assertTrue(parent_hash_mismatch("aaa", "bbb"))
        self.assertFalse(parent_hash_mismatch("aaa", "aaa"))

    def test_spawn_cameras_are_episode1_54(self):
        spawn = spawn_camera_records()
        self.assertEqual(len(spawn), 54)
        spec = episode_spec(1)
        for rec in spawn:
            self.assertEqual(rec["elevation_deg"], spec["elevation_deg"])
            self.assertEqual(rec["radius"], spec["radius"])
        cameras = [base_camera(rec, data_device="cpu") for rec in spawn[:2]]
        zooms = [zoom_camera(rec, data_device="cpu") for rec in spawn[:2]]
        for rec, camera, zoom in zip(spawn[:2], cameras, zooms):
            lookat = camera.camera_center.new_tensor(rec["lookat"])
            self.assertAlmostEqual(
                float((camera.camera_center - lookat).norm().item()),
                rec["radius"],
                places=3,
            )
            self.assertEqual((zoom.image_width, zoom.image_height), (2048, 2048))
        pair = [item for item in training_items(1) if item["pose_id"] == spawn[0]["pose_id"]]
        self.assertEqual(pair[0]["uid_1x"], pair[1]["uid_1x"])
        left = base_camera(pair[0], data_device="cpu")
        right = base_camera(pair[1], data_device="cpu")
        self.assertEqual(left.image_name, right.image_name)
        self.assertNotEqual(sample_seeds(generation_seed=4001, episode=1, pose_index=0, sample_id=0),
                            sample_seeds(generation_seed=4001, episode=1, pose_index=0, sample_id=1))
        self.assertAlmostEqual(EXPECTED_VISITS_PER_IMAGE, 5000 / 108)


if __name__ == "__main__":
    unittest.main()
