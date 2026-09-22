#!/usr/bin/env python3
"""CPU checks for the JAX_068 9-center × 6-azimuth orbit lock."""

from __future__ import annotations

import math
import unittest

from lod.episode1 import IMAGE_SIZE, LOOKAT, orbit_eye
from lod.orbit_9c6a import (
    AZIMUTHS_DEG,
    CENTER_STEPS,
    FULL_L1_CKPT_STEPS,
    FULL_L1_DENSIFY_UNTIL,
    FULL_L1_MAX_POINTS,
    FULL_L1_STEPS,
    FULL_L1_VISITS_EXPECTED,
    DIAGNOSTIC_AZIMUTHS_DEG,
    MIX_RATIO,
    N_BASE_POSES,
    OUTPUT_ROOT,
    PREFERRED_NEIGHBOR_DELTAS_DEG,
    aabb_iou,
    all_base_poses,
    assert_writable_output,
    camera_payload,
    expected_visits,
    experiment_lock,
    ground_quad,
    lock_neighbor,
    locked_archive_hit,
    lookats,
    neighbor_candidates,
    occupancy_union,
    origin_poses,
    pose_id,
    pose_index,
    pose_name,
)
from lod.path import add_gz_src


class Orbit9c6aSpecTests(unittest.TestCase):
    def test_54_unique_training_poses(self) -> None:
        poses = all_base_poses()
        self.assertEqual(len(poses), N_BASE_POSES)
        self.assertEqual(N_BASE_POSES, 54)
        self.assertEqual(len({row["pose_id"] for row in poses}), 54)
        self.assertEqual(len({row["name"] for row in poses}), 54)
        self.assertEqual(len({row["uid_1x"] for row in poses}), 54)
        self.assertEqual([tuple(row["lookat"]) for row in poses if row["index"] == 0][0], lookats()[0])
        origin = origin_poses()
        self.assertEqual(len(origin), 6)
        self.assertEqual([row["azimuth_deg"] for row in origin], list(AZIMUTHS_DEG))
        self.assertEqual(pose_index((0.0, 0.0, 0.0), 0.0), origin[0]["index"])

    def test_lookats_match_official_idu_interior_grid(self) -> None:
        self.assertEqual(
            lookats(),
            (
                (-128.0, -128.0, 0.0),
                (0.0, -128.0, 0.0),
                (128.0, -128.0, 0.0),
                (-128.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (128.0, 0.0, 0.0),
                (-128.0, 128.0, 0.0),
                (0.0, 128.0, 0.0),
                (128.0, 128.0, 0.0),
            ),
        )

    def test_eye_stays_radius_300_from_each_lookat(self) -> None:
        import numpy as np

        for lookat in lookats():
            for azimuth in AZIMUTHS_DEG:
                eye = orbit_eye(azimuth, lookat=lookat)
                self.assertAlmostEqual(float(np.linalg.norm(eye - np.asarray(lookat))), 300.0, places=6)

    def test_lookat_stays_image_center_after_focal_zoom(self) -> None:
        for lookat in ((0.0, 0.0, 0.0), (128.0, -128.0, 0.0)):
            for factor in (1.0, 4.0):
                payload = camera_payload(lookat, 60.0, factor, uid=1)
                proj = payload["lookat_projection"]
                self.assertTrue(proj["in_frame"], msg=(lookat, factor))
                self.assertAlmostEqual(proj["u"], IMAGE_SIZE / 2.0, delta=1.5)
                self.assertAlmostEqual(proj["v"], IMAGE_SIZE / 2.0, delta=1.5)
                self.assertGreater(proj["camera_z"], 0.0)
                extra = camera_payload(lookat, 60.0, 8.0, uid=2)
                self.assertEqual(extra["T"], payload["T"])
                self.assertEqual(extra["R"], payload["R"])

    def test_neighbors_are_same_center_pm10_not_training_cameras(self) -> None:
        for azimuth in AZIMUTHS_DEG:
            cands = neighbor_candidates(azimuth)
            self.assertEqual([row["delta_deg"] for row in cands], list(PREFERRED_NEIGHBOR_DELTAS_DEG))
            for row in cands:
                self.assertNotIn(row["azimuth_deg"], AZIMUTHS_DEG)
                self.assertNotIn(row["azimuth_deg"], DIAGNOSTIC_AZIMUTHS_DEG)
                self.assertNotIn(row["azimuth_deg"], set(DIAGNOSTIC_AZIMUTHS_DEG))

    def test_lock_neighbor_rejects_60deg_training_camera(self) -> None:
        locked = lock_neighbor(
            [
                {"azimuth_deg": 60.0, "delta_deg": 60.0, "coverage": 0.9, "reverse_coverage": 0.9},
                {"azimuth_deg": 350.0, "delta_deg": -10.0, "coverage": 0.2, "reverse_coverage": 0.18},
                {"azimuth_deg": 10.0, "delta_deg": 10.0, "coverage": 0.4, "reverse_coverage": 0.35},
            ]
        )
        self.assertIsNone(locked["fallback"])
        self.assertEqual(locked["selected"]["azimuth_deg"], 10.0)
        self.assertEqual(locked["reason"], "same_center_pm10")
        self.assertTrue(any(item["azimuth_deg"] == 60.0 for item in locked["forbidden"]))

    def test_lock_neighbor_falls_back_when_coverage_is_low(self) -> None:
        locked = lock_neighbor(
            [
                {"azimuth_deg": 350.0, "delta_deg": -10.0, "coverage": 0.01, "reverse_coverage": 0.01},
                {"azimuth_deg": 10.0, "delta_deg": 10.0, "coverage": 0.02, "reverse_coverage": 0.02},
            ]
        )
        self.assertEqual(locked["fallback"], "target_only")
        self.assertEqual(locked["reason"], "low_reprojection_coverage")
        self.assertEqual(locked["selected"]["azimuth_deg"], 10.0)

    def test_4x_adjacent_centers_do_not_cover_the_scene(self) -> None:
        origin = ground_quad((0.0, 0.0, 0.0), 0.0, 4.0)
        east = ground_quad((128.0, 0.0, 0.0), 0.0, 4.0)
        self.assertLess(aabb_iou(origin, east), 0.05)
        union = occupancy_union(
            [ground_quad(lookat, 0.0, 4.0) for lookat in lookats()],
            bounds=(-256.0, 256.0, -256.0, 256.0),
            resolution=8.0,
        )
        self.assertLess(union["coverage_frac"], 0.55)
        self.assertGreater(union["coverage_frac"], 0.15)

    def test_same_center_azimuths_overlap_at_4x(self) -> None:
        a = ground_quad((0.0, 0.0, 0.0), 0.0, 4.0)
        b = ground_quad((0.0, 0.0, 0.0), 60.0, 4.0)
        self.assertGreater(aabb_iou(a, b), 0.5)

    def test_six_view_gets_fewer_visits_per_image(self) -> None:
        single = expected_visits(1)
        six = expected_visits(6)
        self.assertEqual(single["steps"], CENTER_STEPS)
        self.assertEqual(single["mix_ratio"], MIX_RATIO)
        self.assertAlmostEqual(single["expected_visits_per_target"], 400.0)
        self.assertAlmostEqual(six["expected_visits_per_target"], 400.0 / 6.0)
        self.assertLess(six["expected_visits_per_target"], single["expected_visits_per_target"])

    def test_full54_budget_is_separate_and_locked(self) -> None:
        self.assertEqual(FULL_L1_STEPS, 4000)
        self.assertEqual(FULL_L1_MAX_POINTS, 450000)
        self.assertEqual(FULL_L1_CKPT_STEPS, (0, 50, 500, 1000, 2000, 4000))
        self.assertEqual(FULL_L1_DENSIFY_UNTIL, 2000)
        self.assertEqual(FULL_L1_VISITS_EXPECTED, 59)
        lock = experiment_lock()["full_54_l1"]
        self.assertTrue(lock["locked"])
        self.assertTrue(lock["single_shared_l1"])
        self.assertTrue(lock["no_8x"])
        self.assertEqual(lock["test_views_not_for_selection"], ["JAX_068_003_RGB", "JAX_068_012_RGB"])

    def test_refuses_locked_archives(self) -> None:
        self.assertEqual(
            str(locked_archive_hit("skyfall-gs_exp/lod_jax068_scene/joint_l1")),
            "skyfall-gs_exp/lod_jax068_scene",
        )
        self.assertEqual(
            str(locked_archive_hit("skyfall-gs_exp/lod_jax068_scene_80k/joint_l1")),
            "skyfall-gs_exp/lod_jax068_scene_80k",
        )
        self.assertEqual(
            str(locked_archive_hit("skyfall-gs_exp/lod_jax068_episode1/train_l1")),
            "skyfall-gs_exp/lod_jax068_episode1",
        )
        with self.assertRaises(ValueError):
            assert_writable_output("skyfall-gs_exp/lod_jax068_scene/x")
        assert_writable_output(OUTPUT_ROOT / "coverage")
        self.assertTrue(pose_name(LOOKAT, 0.0).startswith("orbit9c6a_"))
        self.assertIn("c0_0_az0", pose_id(LOOKAT, 0.0))

    def test_vendor_accepts_54_named_1_to_4_stage(self) -> None:
        add_gz_src()
        from gaussianzoom_lod.stages import capture_stage, validate_stage_records

        def cam(name: str, fx: float):
            w2c = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 4.0], [0.0, 0.0, 0.0, 1.0]]
            return {"name": name, "width": 32, "height": 32, "fx": fx, "fy": fx, "cx": 16.0, "cy": 16.0, "w2c": w2c}

        names = [row["name"] for row in all_base_poses()]
        records = [
            capture_stage([cam(name, 16.0) for name in names], scale=1.0),
            {"scale": 4.0, "cameras": [cam(name, 64.0) for name in names]},
        ]
        validate_stage_records(records, 2.0)


if __name__ == "__main__":
    unittest.main()
