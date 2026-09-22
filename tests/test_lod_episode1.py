#!/usr/bin/env python3
"""CPU checks for Episode 1 pose, FoV, and nonuniform LoD stage intervals."""

from __future__ import annotations

import math
import unittest

from lod.episode1 import (
    AZIMUTH_DEG,
    BASE_FOV_DEG,
    CENTER_ROI,
    ELEVATION_DEG,
    IMAGE_SIZE,
    LOOKAT,
    MODEL_STEP_SCALE,
    OUTPUT_ROOT,
    PARENT_SH_OUTPUT_ROOT,
    PARENT_FULL_OUTPUT_ROOT,
    RADIUS,
    SCENE_ARCHIVE,
    STAGE_SCALES,
    ZOOM_FACTORS,
    assert_not_original_episode1_archive,
    assert_not_scene_archive,
    camera_name,
    camera_payload,
    expected_fovs_deg,
    horizontal_fov_deg,
    orbit_c2w,
    orbit_eye,
    parent_max_level_for_zoom,
    project_lookat,
    zoom_intrinsics,
)
from lod.path import add_gz_src
from utils.zoom_camera import zoom_fov


class Episode1SpecTests(unittest.TestCase):
    def test_zoom_fov_matches_locked_table(self) -> None:
        fovs = expected_fovs_deg()
        self.assertAlmostEqual(fovs["1x"], 60.0, places=5)
        self.assertAlmostEqual(fovs["4x"], 16.428, places=2)
        self.assertAlmostEqual(fovs["8x"], 8.256, places=2)
        base = math.radians(BASE_FOV_DEG)
        for factor in ZOOM_FACTORS:
            self.assertAlmostEqual(
                horizontal_fov_deg(zoom_factor=factor),
                math.degrees(zoom_fov(base, factor)),
                places=7,
            )

    def test_orbit_looks_at_origin_from_elevation_85_radius_300(self) -> None:
        eye = orbit_eye(AZIMUTH_DEG)
        self.assertAlmostEqual(float(np_norm(eye - LOOKAT)), RADIUS, places=6)
        self.assertAlmostEqual(float(eye[2] / RADIUS), math.sin(math.radians(ELEVATION_DEG)), places=6)
        c2w = orbit_c2w(AZIMUTH_DEG)
        self.assertTrue((c2w[:3, 3] == eye).all())

    def test_lookat_stays_in_frame_after_focal_zoom(self) -> None:
        for factor in ZOOM_FACTORS:
            payload = camera_payload(AZIMUTH_DEG, factor, uid=9000)
            proj = payload["lookat_projection"]
            self.assertTrue(proj["in_frame"], msg=factor)
            self.assertAlmostEqual(proj["u"], IMAGE_SIZE / 2.0, delta=1.5)
            self.assertAlmostEqual(proj["v"], IMAGE_SIZE / 2.0, delta=1.5)
            self.assertGreater(proj["camera_z"], 0.0)

    def test_center_roi_is_legal_at_8x(self) -> None:
        CENTER_ROI.validate_for_zoom(8.0)
        intra = zoom_intrinsics(8.0)
        self.assertAlmostEqual(intra["cx_ndc"], 0.0, places=7)
        self.assertAlmostEqual(intra["fx"] / zoom_intrinsics(1.0)["fx"], 8.0, places=6)

    def test_camera_names_are_unique_on_the_orbit(self) -> None:
        names = [camera_name(az) for az in (0.0, -10.0, 10.0, -20.0)]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(camera_name(0.0).startswith("episode1_"))

    def test_output_dir_is_independent_of_scene_archive(self) -> None:
        self.assertNotEqual(OUTPUT_ROOT.resolve(), SCENE_ARCHIVE.resolve())
        assert_not_scene_archive(OUTPUT_ROOT / "baseline")
        self.assertNotEqual(PARENT_SH_OUTPUT_ROOT.resolve(), OUTPUT_ROOT.resolve())
        self.assertNotEqual(PARENT_SH_OUTPUT_ROOT.resolve(), SCENE_ARCHIVE.resolve())
        assert_not_original_episode1_archive(PARENT_SH_OUTPUT_ROOT / "sh")
        self.assertNotEqual(PARENT_FULL_OUTPUT_ROOT.resolve(), OUTPUT_ROOT.resolve())
        assert_not_original_episode1_archive(PARENT_FULL_OUTPUT_ROOT / "train_l1")
        with self.assertRaises(ValueError):
            assert_not_original_episode1_archive(OUTPUT_ROOT / "train_l1")
        with self.assertRaises(ValueError):
            assert_not_scene_archive(SCENE_ARCHIVE / "anything")

    def test_stage_scales_are_not_a_global_step_of_4(self) -> None:
        self.assertEqual(MODEL_STEP_SCALE, 2.0)
        self.assertEqual(STAGE_SCALES, (1.0, 4.0, 8.0))
        adjacent = [STAGE_SCALES[i] / STAGE_SCALES[i - 1] for i in range(1, len(STAGE_SCALES))]
        self.assertEqual(adjacent, [4.0, 2.0])

    def test_parent_max_level_matches_absorbing_layer(self) -> None:
        self.assertEqual(parent_max_level_for_zoom(4.0), 0)
        self.assertEqual(parent_max_level_for_zoom(8.0), 1)


class Episode1StageValidationTests(unittest.TestCase):
    def test_vendor_accepts_measured_1_4_8_with_step_scale_2(self) -> None:
        add_gz_src()
        from gaussianzoom_lod.stages import capture_stage, validate_stage_records

        def cam(name: str, fx: float):
            w2c = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 4.0], [0.0, 0.0, 0.0, 1.0]]
            return {
                "name": name, "width": 32, "height": 32,
                "fx": fx, "fy": fx, "cx": 16.0, "cy": 16.0, "w2c": w2c,
            }

        records = [
            capture_stage([cam("episode1_az0", 16.0)], scale=1.0),
            {"scale": 4.0, "cameras": [cam("episode1_az0", 64.0)]},
            {"scale": 8.0, "cameras": [cam("episode1_az0", 128.0)]},
        ]
        validate_stage_records(records, 2.0)
        with self.assertRaises(ValueError):
            validate_stage_records(
                [
                    records[0],
                    {"scale": 4.0, "cameras": [cam("episode1_az0", 64.0)]},
                    {"scale": 16.0, "cameras": [cam("episode1_az0", 128.0)]},
                ],
                2.0,
            )


def np_norm(value) -> float:
    import numpy as np

    return float(np.linalg.norm(value))


if __name__ == "__main__":
    unittest.main()
